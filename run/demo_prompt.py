import gradio as gr
import json
import random
from dotenv import load_dotenv
import argparse

load_dotenv()
from zhipuai import ZhipuAI
import os
from tianji import TIANJI_PATH

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Launch Gradio application')
parser.add_argument('--listen', action='store_true', help='Specify to listen on 0.0.0.0')
parser.add_argument('--port', type=int, default=None, help='The port the server should listen on')
parser.add_argument('--root_path', type=str, default=None, help='The root path of the server')
args = parser.parse_args()

file_path = os.path.join(TIANJI_PATH, "tianji/prompt/yiyan_prompt/all_yiyan_prompt.json")
API_KEY = os.environ["ZHIPUAI_API_KEY"]
CHOICES = ["敬酒", "请客", "送礼", "送祝福", "人际交流", "化解尴尬", "矛盾应对"]

with open(file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)


def get_names_by_id(id):
    names = []
    for item in json_data:
        if "id" in item and item["id"] == id:
            names.append(item["name"])

    return list(set(names))  # Remove duplicates


def get_system_prompt_by_name(name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    """Returns the system prompt for the specified name."""
    for item in data:
        if item["name"] == name:
            return item["system_prompt"]
    return None  # If the name is not found


def change_example(name, cls_choose_value, chatbot):
    now_example = []
    # 直接将 chatbot 设置为 None 或空列表来触发清除
    chatbot = []
    for i in cls_choose_value:
        if i["name"] == name:
            now_example = [[j["input"], j["output"]] for j in i["example"]]
    if not now_example:
        raise gr.Error("获取example出错！")
    return gr.update(samples=now_example), [] # 返回空历史


def random_button_click(chatbot):
    choice_number = random.randint(0, 6)
    now_id = choice_number + 1
    cls_choose = CHOICES[choice_number]
    now_json_data = _get_id_json_id(choice_number)
    random_name = [i["name"] for i in now_json_data]
    if chatbot is not None:
        print("切换场景清理bot历史")
        chatbot.clear()
    return (
        cls_choose,
        now_json_data,
        gr.update(choices=get_names_by_id(now_id), value=random.choice(random_name)),
    )


def example_click(dataset, name, now_json):
    system = ""
    for i in now_json:
        if i["name"] == name:
            system = i["system_prompt"]

    if system_prompt == "":
        print(name, now_json)
        raise "遇到代码问题，清重新选择场景"
    return dataset[0], system


def _get_id_json_id(idx):
    now_id = idx + 1  # index + 1
    now_id_json_data = []
    for item in json_data:
        if int(item["id"]) == int(now_id):
            temp_dict = dict(
                name=item["name"],
                example=item["example"],
                system_prompt=item["system_prompt"],
            )
            now_id_json_data.append(temp_dict)
    return now_id_json_data


def cls_choose_change(idx):
    now_id = idx + 1
    return _get_id_json_id(idx), gr.update(
        choices=get_names_by_id(now_id), value=get_names_by_id(now_id)[0]
    )


def combine_message_and_history(message, chat_history):
    # 将聊天历史中的每个元素（假设是元组）转换为字符串
    history_str = "\n".join(f"{sender}: {text}" for sender, text in chat_history)

    # 将新消息和聊天历史结合成一个字符串
    full_message = f"{history_str}\nUser: {message}"
    return full_message


def respond(system_prompt, message, chat_history):
    # 兼容性处理：如果此时 chat_history 是 None，初始化为空列表
    if chat_history is None:
        chat_history = []

    if len(chat_history) > 11:
        chat_history.clear()
        # 注意：这里也需要改为字典格式
        chat_history.append({"role": "assistant", "content": "对话超过限制，已重新开始"})

    # 由于最新版 chat_history 是字典列表，我们需要修改 combine 函数的逻辑或在这里手动提取
    # 为了最小化改动，我们直接在这里处理发送给 API 的消息
    history_for_api = []
    for msg in chat_history:
        # 从字典中提取文本用于 API 拼接（如果需要的话）
        role = "User" if msg["role"] == "user" else "Bot"
        history_for_api.append((role, msg["content"]))

    message1 = combine_message_and_history(message, history_for_api)

    client = ZhipuAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message1},
        ],
    )

    bot_message_text = response.choices[0].message.content

    # 核心修改：将消息以字典格式追加到聊天历史中
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message_text})

    return "", chat_history


def clear_history(chat_history):
    return [] # 直接返回空列表清空


def regenerate(chat_history, system_prompt):
    if chat_history and len(chat_history) >= 2:
        # 字典格式下，最后一条是助手，倒数第二条是用户
        last_user_message = chat_history[-2]["content"]
        # 移除最后两条（旧的用户输入和旧的助手回复）
        chat_history = chat_history[:-2]
        return respond(system_prompt, last_user_message, chat_history)
    return "", chat_history


TITLE = """
# Tianji 人情世故大模型系统——prompt版\n
## 💫基于开源项目https://github.com/SocialAI-tianji/Tianji
## 我们支持不同模型进行对话，你可以选择你喜欢的模型进行对话。
## 使用方法：选择或随机一个场景，输入提示词（或者点击上面的Example自动填充），随后发送！
"""

with gr.Blocks() as demo:
    chat_history = gr.State()
    now_json_data = gr.State(value=_get_id_json_id(0))
    now_name = gr.State()
    gr.Markdown(TITLE)
    cls_choose = gr.Radio(label="请选择任务大类", choices=CHOICES, type="index", value="敬酒")
    input_example = gr.Dataset(
        components=["text", "text"],
        samples=[
            ["请先选择合适的场景", "请先选择合适的场景"],
        ],
    )
    with gr.Row():
        with gr.Column(scale=1):
            dorpdown_name = gr.Dropdown(
                choices=get_names_by_id(1),
                label="场景",
                info="请选择合适的场景",
                interactive=True,
            )
            system_prompt = gr.TextArea(label="系统提示词")  # TODO 需要给初始值嘛？包括example
            random_button = gr.Button("🪄点我随机一个试试！", size="lg")
            dorpdown_name.change(
                fn=get_system_prompt_by_name,
                inputs=[dorpdown_name],
                outputs=[system_prompt],
            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="聊天界面",
                # 核心修改：将原来的 [["user", "assistant"]] 改为 {"role": ..., "content": ...}
                value=[]
            )
            msg = gr.Textbox(label="输入信息")
            msg.submit(
                respond, inputs=[system_prompt, msg, chatbot], outputs=[msg, chatbot]
            )
            submit = gr.Button("发送").click(
                respond, inputs=[system_prompt, msg, chatbot], outputs=[msg, chatbot]
            )
            with gr.Row():
                clear = gr.Button("清除历史记录").click(
                    clear_history, inputs=[chatbot], outputs=[chatbot]
                )
                regenerate = gr.Button("重新生成").click(
                    regenerate, inputs=[chatbot, system_prompt], outputs=[msg, chatbot]
                )

    cls_choose.change(
        fn=cls_choose_change, inputs=cls_choose, outputs=[now_json_data, dorpdown_name]
    )
    dorpdown_name.change(
        fn=change_example,
        inputs=[dorpdown_name, now_json_data, chatbot],
        outputs=[input_example, chat_history],
    )
    input_example.click(
        fn=example_click,
        inputs=[input_example, dorpdown_name, now_json_data],
        outputs=[msg, system_prompt],
    )
    random_button.click(
        fn=random_button_click,
        inputs=chatbot,
        outputs=[cls_choose, now_json_data, dorpdown_name],
    )

if __name__ == "__main__":
    server_name = '0.0.0.0' if args.listen else None
    server_port = args.port
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,  # 暂时关闭 share，避免 frpc 和连接超时报错
        show_error = True
    )
