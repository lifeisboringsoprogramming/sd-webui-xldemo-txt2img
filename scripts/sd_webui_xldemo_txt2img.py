import gradio as gr
from modules.shared import opts, OptionInfo
from modules import script_callbacks
from xldemo_txt2img_ui import make_ui
from xldemo_txt2img import XLDEMO_MODEL_CHOICES


def on_ui_tabs():
    return [(make_ui(), "SDXL Demo", "xldemo_txt2img")]


def on_ui_settings():
    section = ("xldemo_txt2img", "SDXL Demo")

    opts.add_option(
        "xldemo_txt2img_huggingface_access_token", OptionInfo(
            "", "Huggingface access token (Restart WebUI to take effect)", section=section)
    )
    opts.add_option(
        "xldemo_txt2img_model", OptionInfo(XLDEMO_MODEL_CHOICES[0], "Model (Restart WebUI to take effect)", gr.Dropdown, lambda: {
                                           "choices": XLDEMO_MODEL_CHOICES}, section=section)
    )

    opts.add_option(
        "xldemo_txt2img_load_refiner_on_startup", OptionInfo(
            True, "Enable refiner (Restart WebUI to take effect)", section=section)
    )


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
