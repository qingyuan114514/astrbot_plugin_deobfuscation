import os
import io
import glob
import asyncio
import functools
import uuid
import socket
import ipaddress
from urllib.parse import urlparse
import aiohttp
import numpy as np
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api.star import StarTools

from .tomato_scramble import TomatoScramble


# 安全限制常量
_MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20MB 下载体积上限
_MAX_PIXEL_COUNT = 4096 * 4096          # ~16M 像素上限

# 启用 PIL 解压炸弹防护
PILImage.MAX_IMAGE_PIXELS = _MAX_PIXEL_COUNT


@register("tomato_scramble", "清远", "小番茄图片混淆/解混淆插件", "1.2.0")
class TomatoScramblePlugin(Star):

    def __init__(self, context: Context, config=None):
        super().__init__(context, config)
        self.data_dir = str(StarTools.get_data_dir("tomato_scramble"))
        self._session: aiohttp.ClientSession | None = None

        # 读取配置
        self.use_forward = False
        self.forward_sender_name = "解混淆结果"
        if config:
            self.use_forward = config.get("use_forward_message", False)
            self.forward_sender_name = config.get("forward_sender_name", "解混淆结果")

    @staticmethod
    async def _validate_url(url: str) -> None:
        """校验 URL 安全性，防止 SSRF 攻击"""
        parsed = urlparse(url)

        # 只允许 http/https 协议
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"不允许的协议: {parsed.scheme}")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("无法解析主机名")

        # 异步 DNS 解析，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        try:
            addr_infos = await loop.getaddrinfo(hostname, None)
        except socket.gaierror:
            raise ValueError(f"无法解析主机: {hostname}")

        for info in addr_infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError(f"禁止访问内网/保留地址: {ip}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """懒创建并复用 aiohttp.ClientSession"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    async def _download_image(self, url: str) -> bytes:
        await self._validate_url(url)
        session = await self._get_session()
        async with session.get(url, allow_redirects=False) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            # 优先通过 Content-Length 快速拒绝超大文件
            content_length = resp.content_length
            if content_length and content_length > _MAX_DOWNLOAD_SIZE:
                raise ValueError(f"图片体积过大: {content_length / 1024 / 1024:.1f}MB，上限 20MB")
            # 流式读取，防止实际数据超限
            chunks = []
            total = 0
            async for chunk in resp.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > _MAX_DOWNLOAD_SIZE:
                    raise ValueError("图片体积超过 20MB 上限")
                chunks.append(chunk)
            return b"".join(chunks)

    def _extract_images(self, event: AstrMessageEvent) -> list:
        """从消息中提取图片，支持直接附带图片和引用消息中的图片"""
        images = []

        for comp in event.message_obj.message:
            if isinstance(comp, Comp.Image):
                images.append(comp)

        # 如果消息本身没有图片，尝试从引用的消息中提取
        if not images:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Reply) and comp.chain:
                    for chain_comp in comp.chain:
                        if isinstance(chain_comp, Comp.Image):
                            images.append(chain_comp)
                    break  # 只处理第一个引用

        return images

    def _get_image_url(self, img_comp) -> str:
        if hasattr(img_comp, 'url') and img_comp.url:
            return img_comp.url
        if hasattr(img_comp, 'file') and img_comp.file:
            return img_comp.file
        raise Exception("cannot get image url")

    def _process_image_sync(self, img_bytes: bytes, mode: str = "decrypt", key: float = 1.0) -> str:
        """CPU 密集的同步处理逻辑，由 _process_image 在线程池中调用"""
        img = PILImage.open(io.BytesIO(img_bytes))
        width, height = img.size
        if width * height > _MAX_PIXEL_COUNT:
            raise ValueError(f"图片像素过大: {width}x{height} = {width * height}，上限 {_MAX_PIXEL_COUNT}")
        img = img.convert("RGB")
        pixels = list(img.getdata())

        scrambler = TomatoScramble(pixels, width, height, key)
        if mode == "encrypt":
            new_pixels = scrambler.encrypt()
        else:
            new_pixels = scrambler.decrypt()

        new_img = PILImage.fromarray(new_pixels.reshape(height, width, 3).astype(np.uint8), "RGB")

        filename = f"{mode}_{uuid.uuid4().hex}.png"
        save_path = os.path.join(self.data_dir, filename)
        new_img.save(save_path, "PNG")
        return save_path

    async def _process_image(self, img_bytes: bytes, mode: str = "decrypt", key: float = 1.0) -> str:
        """将 CPU 密集操作放到线程池执行，避免阻塞事件循环"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self._process_image_sync, img_bytes, mode, key)
        )

    def _get_bot_uin(self, event: AstrMessageEvent) -> str:
        """尝试获取机器人的QQ号，用于合并转发节点"""
        try:
            return str(event.message_obj.self_id)
        except Exception:
            return "0"

    async def _do_process(self, event, images, mode, key, mode_text):
        """处理图片，根据配置决定是逐张发送还是合并转发"""
        success_paths = []
        errors = []
        for i, img_comp in enumerate(images):
            try:
                url = self._get_image_url(img_comp)
            except Exception as e:
                logger.error(f"{mode_text} 第{i+1}张: 获取图片URL失败 - {type(e).__name__}: {e}")
                errors.append((i + 1, f"获取URL失败: {e}"))
                continue
            try:
                img_bytes = await self._download_image(url)
            except ValueError as e:
                logger.warning(f"{mode_text} 第{i+1}张: 安全校验拦截 - {e}")
                errors.append((i + 1, f"安全校验: {e}"))
                continue
            except aiohttp.ClientError as e:
                logger.error(f"{mode_text} 第{i+1}张: 网络请求失败 - {type(e).__name__}: {e}")
                errors.append((i + 1, f"下载失败: {e}"))
                continue
            except Exception as e:
                logger.error(f"{mode_text} 第{i+1}张: 下载异常 - {type(e).__name__}: {e}")
                errors.append((i + 1, f"下载异常: {e}"))
                continue
            try:
                save_path = await self._process_image(img_bytes, mode, key)
                success_paths.append(save_path)
            except ValueError as e:
                logger.warning(f"{mode_text} 第{i+1}张: 图片校验不通过 - {e}")
                errors.append((i + 1, f"图片校验: {e}"))
            except Exception as e:
                logger.error(f"{mode_text} 第{i+1}张: 处理失败 - {type(e).__name__}: {e}", exc_info=True)
                errors.append((i + 1, f"处理失败: {e}"))

        if self.use_forward and success_paths:
            # 合并转发模式：所有图片放进同一个 Node 的 content 里
            bot_uin = self._get_bot_uin(event)
            content = [Comp.Image.fromFileSystem(p) for p in success_paths]
            if errors:
                err_text = "\n".join(f"第{idx}张处理失败: {err}" for idx, err in errors)
                content.append(Comp.Plain(err_text))
            node = Comp.Node(
                uin=bot_uin,
                name=self.forward_sender_name,
                content=content
            )
            yield event.chain_result([node])
        else:
            # 逐张发送模式
            for save_path in success_paths:
                yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            for idx, err in errors:
                yield event.plain_result(f"第{idx}张处理失败: {err}")

    @filter.command("解析", alias=["解混淆"])
    async def decrypt_cmd(self, event: AstrMessageEvent, key: float = 1.0):
        if key <= 0 or key >= 1.618:
            yield event.plain_result("密钥范围需要在 (0, 1.618) 之间")
            event.stop_event()
            return

        images = self._extract_images(event)

        if images:
            yield event.plain_result("收到，正在解混淆...")
            event.stop_event()
            async for result in self._do_process(event, images, "decrypt", key, "解混淆"):
                yield result
        else:
            yield event.plain_result("请在指令消息中附带或引用需要解混淆的图片~")
            event.stop_event()

    @filter.command("混淆", alias=["加密"])
    async def encrypt_cmd(self, event: AstrMessageEvent, key: float = 1.0):
        if key <= 0 or key >= 1.618:
            yield event.plain_result("密钥范围需要在 (0, 1.618) 之间")
            event.stop_event()
            return

        images = self._extract_images(event)

        if images:
            yield event.plain_result("收到，正在混淆...")
            event.stop_event()
            async for result in self._do_process(event, images, "encrypt", key, "混淆"):
                yield result
        else:
            yield event.plain_result("请在指令消息中附带或引用需要混淆的图片~")
            event.stop_event()

    async def terminate(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        for f in glob.glob(os.path.join(self.data_dir, "*.png")):
            try:
                os.remove(f)
            except PermissionError:
                logger.warning(f"清理缓存文件权限不足，跳过: {f}")
            except OSError as e:
                logger.warning(f"清理缓存文件失败: {f} - {type(e).__name__}: {e}")
