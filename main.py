import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "SiliconFlow 手办化/图生图插件",
    "1.3.5",
    "https://github.com/shkjw/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url:
                logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url

        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在尝试下载图片: {url}")
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=60) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as e:
                logger.error(f"图片下载失败: {e}", exc_info=True)
                return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit():
                return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception:
                pass
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw:
                return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                                img_bytes_list.append(img)
                            elif s_chain.file and (img := await self._load_bytes(s_chain.file)):
                                img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url and (img := await self._load_bytes(seg.url)):
                        img_bytes_list.append(img)
                    elif seg.file and (img := await self._load_bytes(seg.file)):
                        img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                return img_bytes_list

            return img_bytes_list

        async def terminate(self):
            if self.session and not self.session.closed:
                await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()

        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"

        self.user_counts: Dict[str, int] = {}
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}

        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)

        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()

        logger.info("FigurinePro (SiliconFlow版) 插件已加载")

        # 检查 Key 配置
        model_list = self.conf.get("model_list", [])
        pool_keys = self.conf.get("api_keys", [])
        has_custom_keys = any(m.get("key") for m in model_list if isinstance(m, dict))

        if not pool_keys and not has_custom_keys:
            logger.warning("FigurinePro: 未配置任何 API Key，插件可能无法生图")

    async def _load_prompt_map(self):
        self.prompt_map.clear()

        # 1. 内置基础指令映射 (确保这些指令被识别)
        base_cmd_map = {
            "手办化": "figurine",
            "Q版化": "q_version",
            "痛屋化": "pain_room",
            "痛车化": "pain_car",
            "cos化": "cos",
            "cos自拍": "cos_selfie",
            "孤独的我": "clown",
            "第一视角": "view_1",
            "第三视角": "view_3",
            "鬼图": "ghost",
        }
        for k in base_cmd_map.keys():
            self.prompt_map[k] = "[内置预设]"

        # 2. 加载配置中的 Prompt 列表
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            if ":" in item:
                key, value = item.split(":", 1)
                self.prompt_map[key.strip()] = value.strip()

        logger.info(f"加载了 {len(self.prompt_map)} 个 prompts。")

    def _get_all_models(self) -> List[str]:
        """从配置的 model_list 中获取所有 model ID"""
        model_list_cfg = self.conf.get("model_list", [])
        models = []
        for item in model_list_cfg:
            if isinstance(item, dict) and item.get("id"):
                models.append(item["id"])
            elif isinstance(item, str):  # 兼容纯字符串配置
                models.append(item)
        return models

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _norm_id(self, raw_id: Any) -> str:
        if raw_id is None:
            return ""
        return str(raw_id).strip()

    @filter.command("切换模型", aliases={"SwitchModel", "模型列表"}, prefix_optional=True)
    async def on_switch_model(self, event: AstrMessageEvent):
        all_models = self._get_all_models()
        raw_msg = event.message_str.strip()
        parts = raw_msg.split()

        if len(parts) == 1:
            current_model = self.conf.get("model", "")
            msg = "📋 **可用模型列表**:\n"
            msg += "------------------\n"
            for idx, model_name in enumerate(all_models):
                seq_num = idx + 1
                status = "✅ (当前)" if model_name == current_model else ""
                msg += f"{seq_num}. {model_name} {status}\n"
            msg += "------------------\n"
            msg += "📝 **指令**:\n1. `#切换模型 <序号>`\n2. `#lm列表` 查看预设"
            yield event.plain_result(msg)
            return

        arg = parts[1]
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以更改全局默认模型。")
            return

        if not arg.isdigit():
            yield event.plain_result("❌ 格式错误。请输入数字序号。")
            return

        target_idx = int(arg) - 1
        if 0 <= target_idx < len(all_models):
            new_model = all_models[target_idx]
            self.conf["model"] = new_model
            try:
                if hasattr(self.conf, "save"):
                    self.conf.save()
            except:
                pass
            yield event.plain_result(f"✅ 切换成功！\n当前默认模型: **{new_model}**")
        else:
            yield event.plain_result(f"❌ 序号无效。")

    @filter.command("lm列表", aliases={"lmlist", "预设列表"}, prefix_optional=True)
    async def on_get_preset_list(self, event: AstrMessageEvent):
        """输出所有可用预设列表"""
        if not self.prompt_map:
            yield event.plain_result("⚠️ 当前没有可用的预设。")
            return

        all_keys = sorted(list(self.prompt_map.keys()))

        msg = "📜 **可用预设列表**\n"
        msg += "==================\n"
        msg += "  " + "、".join(all_keys)
        msg += "\n==================\n"
        msg += "使用方法: #预设名 [图片] 或 #bnn <提示词>"

        yield event.plain_result(msg)

    async def _get_api_key(self, model_name: str) -> str | None:
        # 1. 优先检查模型专用Key
        model_list_cfg = self.conf.get("model_list", [])
        target_model_cfg = next((m for m in model_list_cfg if isinstance(m, dict) and m.get("id") == model_name), None)
        if target_model_cfg and target_model_cfg.get("key"):
            return target_model_cfg["key"]

        # 2. 使用全局池
        keys = self.conf.get("api_keys", [])
        if not keys:
            return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        """从 SiliconFlow 响应中提取图片 URL"""
        try:
            url = data["images"][0]["url"]
            logger.info(f"成功从 API 响应中提取到 URL: {url[:50]}...")
            return url
        except (IndexError, TypeError, KeyError):
            return None

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str,
                        override_model: str | None = None) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url:
            return "API URL 未配置"

        model_name = override_model or self.conf.get("model")
        if not model_name:
            return "模型名称未配置"

        api_key = await self._get_api_key(model_name)
        if not api_key:
            return "无可用的 API Key"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        # 构建 SiliconFlow Payload
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "image_size": "1024x1024",  # 默认尺寸
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5
        }

        # 图生图处理
        if image_bytes_list:
            try:
                img_b64 = base64.b64encode(image_bytes_list[0]).decode("utf-8")
                payload["image"] = f"data:image/png;base64,{img_b64}"
                # 如果 SiliconFlow 模型支持 image2, image3，可在此扩展
            except Exception as e:
                return f"图片编码失败: {e}"

        logger.info(f"发送请求: URL={api_url}, Model={model_name}, HasImage={bool(image_bytes_list)}")

        try:
            if not self.iwf:
                return "ImageWorkflow 未初始化"

            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy,
                                             timeout=120) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return f"API请求失败 (HTTP {resp.status}): {error_text[:200]}"

                data = await resp.json()

                if "images" not in data or not data["images"]:
                    if "error" in data:
                        return data["error"].get("message", json.dumps(data["error"]))
                    return f"API响应异常: {str(data)[:200]}"

                gen_image_url = self._extract_image_url_from_response(data)
                if not gen_image_url:
                    return f"解析URL失败: {str(data)[:200]}"

                return await self.iwf._download_image(gen_image_url) or "下载生成的图片失败"

        except asyncio.TimeoutError:
            return "请求超时"
        except Exception as e:
            logger.error(f"API调用错误: {e}", exc_info=True)
            return f"系统错误: {e}"

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return
        text = event.message_str.strip()
        if not text:
            return

        tokens = text.split()
        if not tokens:
            return

        # 解析指令和可能的序号 (e.g., 手办化(2))
        raw_cmd = tokens[0].strip()
        cmd_token = raw_cmd
        temp_model_idx = None

        match = re.search(r"[\(（](\d+)[\)）]$", raw_cmd)
        if match:
            temp_model_idx = int(match.group(1))
            cmd_token = raw_cmd[:match.start()].strip()

        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False

        if cmd_token == bnn_command:
            user_prompt = " ".join(tokens[1:]).strip()
            is_bnn = True
        elif cmd_token in self.prompt_map:
            val = self.prompt_map.get(cmd_token)
            if val and val != "[内置预设]":
                user_prompt = val
        else:
            return  # 指令不匹配

        # --- 权限检查 ---
        sender_id = self._norm_id(event.get_sender_id())
        group_id = self._norm_id(event.get_group_id())
        is_master = self.is_global_admin(event)

        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return

            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",
                                                                                                   []): return

            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id)

            # 群组限制 (如果开启)
            if group_id and self.conf.get("enable_group_limit", False):
                if group_count <= 0:
                    yield event.plain_result("❌ 本群次数已用尽。")
                    return
            # 个人限制 (如果开启且未被群限制覆盖，或同时生效)
            elif self.conf.get("enable_user_limit", True):
                if user_count <= 0:
                    yield event.plain_result("❌ 您的次数已用尽。")
                    return

        # --- 获取图片 ---
        images_to_process = []
        is_text_to_image = False

        if self.iwf:
            img_bytes_list = await self.iwf.get_images(event)
            if not img_bytes_list:
                if is_bnn:
                    # bnn + 无图 = 文生图
                    if not user_prompt:
                        yield event.plain_result(f"请提供描述。用法: #{bnn_command} <描述>")
                        return
                    is_text_to_image = True
                else:
                    # 预设指令通常需要图片 (图生图)
                    yield event.plain_result("请发送或引用一张图片。")
                    return
            else:
                images_to_process = [img_bytes_list[0]]  # 仅取第一张

        # --- 模型覆盖 ---
        override_model_name = None
        if temp_model_idx is not None:
            all_models = self._get_all_models()
            if 1 <= temp_model_idx <= len(all_models):
                override_model_name = all_models[temp_model_idx - 1]
            else:
                yield event.plain_result(f"⚠️ 指定的模型序号 {temp_model_idx} 无效。")

        display_label = user_prompt[:10] + "..." if len(user_prompt) > 10 else (user_prompt or cmd_token)
        action_type = "文生图" if is_text_to_image else "图生图"
        yield event.plain_result(f"🎨 收到{action_type}请求，正在生成 [{display_label}]...")

        # --- 执行生图 ---
        start_time = datetime.now()
        res = await self._call_api(images_to_process, user_prompt, override_model=override_model_name)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            # 扣费逻辑
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True):
                    await self._decrease_user_count(sender_id)

            caption = f"✅ 生成成功 ({elapsed:.2f}s) | {display_label}"
            if not is_master and self.conf.get("enable_user_limit", True):
                caption += f" | 剩余: {self._get_user_count(sender_id)}"

            yield event.chain_result([Image.fromBytes(res), Plain(caption)])
        else:
            yield event.plain_result(f"❌ 生成失败: {res}")

        event.stop_event()

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        raw = event.message_str.strip()
        cmd_prefix = "lm添加"
        if raw.startswith(cmd_prefix):
            raw = raw[len(cmd_prefix):].strip()

        if ":" not in raw:
            yield event.plain_result('格式错误, 示例: #lm添加 触发词:提示词')
            return

        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])

        found = False
        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break
        if not found:
            prompt_list.append(f"{key}:{new_value}")

        await self.conf.set("prompt_list", prompt_list)
        await self._load_prompt_map()
        yield event.plain_result(f"✅ 已保存预设:\n{key}:{new_value}")

    @filter.command("lm查看", aliases={"lmv", "lm预览"}, prefix_optional=True)
    async def lm_preview_prompt(self, event: AstrMessageEvent):
        raw = event.message_str.strip()
        parts = raw.split()
        if len(parts) < 2:
            yield event.plain_result("用法: #lm查看 <关键词>")
            return

        keyword = parts[1].strip()
        prompt_content = self.prompt_map.get(keyword)

        if prompt_content:
            yield event.plain_result(f"🔍 关键词【{keyword}】的提示词：\n\n{prompt_content}")
        else:
            yield event.plain_result(f"❌ 未找到关键词【{keyword}】的预设。")

    @filter.command("lm帮助", aliases={"lmh", "手办化帮助"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        parts = event.message_str.strip().split()
        keyword = parts[1] if len(parts) > 1 else ""
        if not keyword:
            yield event.plain_result("请指定要查看的预设词，例如：#lm帮助 手办化\n使用 #lm列表 查看所有可用预设。")
            return

        prompt = self.prompt_map.get(keyword)
        content = f"📄 预设 [{keyword}] 内容:\n{prompt}" if prompt else f"❌ 未找到 [{keyword}]"
        yield event.plain_result(content)

    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image(self, event: AstrMessageEvent):
        # 兼容旧指令，直接调用核心逻辑
        prompt = event.message_str.strip()
        if not prompt:
            yield event.plain_result("请提供描述。")
            return

        yield event.plain_result(f"🎨 正在生成: {prompt[:10]}...")
        # 直接调用 API，传空图片列表
        res = await self._call_api([], prompt)

        if isinstance(res, bytes):
            # 这里简单处理，不走通用扣费逻辑（或者根据需要添加）
            yield event.chain_result([Image.fromBytes(res), Plain("✅ 生成成功")])
        else:
            yield event.plain_result(f"❌ 失败: {res}")

    # ================= 统计与存储 =================

    async def _load_user_counts(self):
        if not self.user_counts_file.exists():
            self.user_counts = {}
            return
        try:
            content = await asyncio.to_thread(self.user_counts_file.read_text, "utf-8")
            self.user_counts = json.loads(content)
        except:
            self.user_counts = {}

    async def _save_user_counts(self):
        try:
            data = json.dumps(self.user_counts, indent=4)
            await asyncio.to_thread(self.user_counts_file.write_text, data, "utf-8")
        except:
            pass

    def _get_user_count(self, uid: str) -> int:
        return self.user_counts.get(self._norm_id(uid), 0)

    async def _decrease_user_count(self, uid: str):
        u = self._norm_id(uid)
        c = self._get_user_count(u)
        if c > 0:
            self.user_counts[u] = c - 1
            await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists():
            self.group_counts = {}
            return
        try:
            content = await asyncio.to_thread(self.group_counts_file.read_text, "utf-8")
            self.group_counts = json.loads(content)
        except:
            self.group_counts = {}

    async def _save_group_counts(self):
        try:
            data = json.dumps(self.group_counts, indent=4)
            await asyncio.to_thread(self.group_counts_file.write_text, data, "utf-8")
        except:
            pass

    def _get_group_count(self, gid: str) -> int:
        return self.group_counts.get(self._norm_id(gid), 0)

    async def _decrease_group_count(self, gid: str):
        g = self._norm_id(gid)
        c = self._get_group_count(g)
        if c > 0:
            self.group_counts[g] = c - 1
            await self._save_group_counts()

    async def _load_user_checkin_data(self):
        if not self.user_checkin_file.exists():
            self.user_checkin_data = {}
            return
        try:
            content = await asyncio.to_thread(self.user_checkin_file.read_text, "utf-8")
            self.user_checkin_data = json.loads(content)
        except:
            self.user_checkin_data = {}

    async def _save_user_checkin_data(self):
        try:
            data = json.dumps(self.user_checkin_data, indent=4)
            await asyncio.to_thread(self.user_checkin_file.write_text, data, "utf-8")
        except:
            pass

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 未开启签到。")
            return
        uid = self._norm_id(event.get_sender_id())
        today = datetime.now().strftime("%Y-%m-%d")
        if self.user_checkin_data.get(uid) == today:
            yield event.plain_result(f"已签到。剩余: {self._get_user_count(uid)}")
            return

        reward = int(self.conf.get("checkin_fixed_reward", 3))
        if self.conf.get("enable_random_checkin", False):
            reward = random.randint(1, max(1, int(self.conf.get("checkin_random_reward_max", 5))))

        self.user_counts[uid] = self._get_user_count(uid) + reward
        await self._save_user_counts()
        self.user_checkin_data[uid] = today
        await self._save_user_checkin_data()
        yield event.plain_result(f"🎉 签到成功 +{reward}次。")

    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        text = event.message_str.strip()

        target_uid = None
        count = 0

        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        if at_seg:
            target_uid = str(at_seg.qq)
            match = re.search(r"(\d+)$", text)
            if match: count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", text)
            if match:
                target_uid = match.group(1)
                count = int(match.group(2))

        if target_uid:
            target_uid = self._norm_id(target_uid)
            c = self._get_user_count(target_uid) + count
            self.user_counts[target_uid] = c
            await self._save_user_counts()
            yield event.plain_result(f"✅ 用户 {target_uid} 现剩余 {c} 次")
        else:
            yield event.plain_result("格式错误: #手办化增加用户次数 <QQ号/@用户> <次数>")

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        uid = self._norm_id(event.get_sender_id())
        msg = f"👤 个人剩余: {self._get_user_count(uid)}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self._get_group_count(gid)}"
        yield event.plain_result(msg)

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        keys = event.message_str.strip().split()
        if not keys:
            return
        current = self.conf.get("api_keys", [])
        added = [k for k in keys if k not in current]
        current.extend(added)
        await self.conf.set("api_keys", current)
        yield event.plain_result(f"✅ 已添加 {len(added)} 个Key")

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        keys = self.conf.get("api_keys", [])
        msg = "\n".join([f"{i + 1}. {k[:8]}..." for i, k in enumerate(keys)])
        yield event.plain_result(f"🔑 Key列表:\n{msg}")

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
        param = event.message_str.strip()
        keys = self.conf.get("api_keys", [])
        if param == "all":
            keys = []
        elif param.isdigit():
            idx = int(param) - 1
            if 0 <= idx < len(keys):
                keys.pop(idx)
        await self.conf.set("api_keys", keys)
        yield event.plain_result("✅ 删除完成")

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
