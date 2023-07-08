from modules.shared import opts, OptionInfo
from modules import script_callbacks
from xldemo_txt2img_ui import make_ui


def on_ui_tabs():
    return [(make_ui(), "SDXL 0.9 Demo", "xldemo_txt2img")]


def on_ui_settings():
    section = ("xldemo_txt2img", "SDXL 0.9 Demo")

    opts.add_option(
        "xldemo_txt2img_huggingface_access_token", OptionInfo(
            "", "Huggingface access token", section=section)
    )

    opts.add_option(
        "xldemo_txt2img_load_refiner_on_startup", OptionInfo(
            True, "Enable refiner", section=section)
    )

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)

