import os
import io
import time
import glob
import aiohttp
import numpy as np
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .tomato_scramble import TomatoScramble


@register("tomato_scramble", "清远", "小番茄图片混淆/解混淆插件", "1.2.0")
class TomatoScramblePlugin(Star):

    def __init__(self, context: Context, config=None):
        super().__init__(context, config)
        self.data_dir = os.path.join("data", "plugin_data", "tomato_scramble")
        os.makedirs(self.data_dir, exist_ok=True)

        # 读取配置
        self.use_forward = False
        self.forward_sender_name = "解混淆结果"
        if config:
            self.use_forward = config.get("use_forward_message", False)
            self.forward_sender_name = config.get("forward_sender_name", "解混淆结果")

    async def _download_image(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.read()
                raise Exception(f"HTTP {resp.status}")

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

    def _process_image(self, img_bytes: bytes, mode: str = "decrypt", key: float = 1.0) -> str:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = img.size
        pixels = list(img.getdata())

        scrambler = TomatoScramble(pixels, width, height, key)
        if mode == "encrypt":
            new_pixels = scrambler.encrypt()
        else:
            new_pixels = scrambler.decrypt()

        # numpy 数组直接通过 frombuffer 构建图像，避免 putdata 的逐像素开销
        new_img = PILImage.fromarray(new_pixels.reshape(height, width, 3).astype(np.uint8), "RGB")

        filename = f"{mode}_{int(time.time() * 1000)}.png"
        save_path = os.path.join(self.data_dir, filename)
        new_img.save(save_path, "PNG")
        return save_path

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
                img_bytes = await self._download_image(url)
                save_path = self._process_image(img_bytes, mode, key)
                success_paths.append(save_path)
            except Exception as e:
                logger.error(f"{mode_text} error: {e}")
                errors.append((i + 1, str(e)))

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
        for f in glob.glob(os.path.join(self.data_dir, "*.png")):
            try:
                os.remove(f)
            except Exception:
                pass
