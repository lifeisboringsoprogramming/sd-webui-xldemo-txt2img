import gradio as gr

from modules.shared import opts
from modules.ui_components import ToolButton

from xldemo_txt2img import XLDEMO_HUGGINGFACE_ACCESS_TOKEN, XLDEMO_LOAD_REFINER_ON_STARTUP
from xldemo_txt2img import do_xldemo_txt2img_infer
from xldemo_txt2img_ui_common import create_seed_inputs, create_output_panel, connect_reuse_seed, gr_show


switch_values_symbol = '\U000021C5'  # ⇅


def make_ui():
    id_part = 'xldemo_txt2img'

    if XLDEMO_HUGGINGFACE_ACCESS_TOKEN is None or XLDEMO_HUGGINGFACE_ACCESS_TOKEN.strip() == '':
        with gr.Blocks(analytics_enabled=False) as ui_component:
            gr.HTML(value="""<div style='font-size: 1.4em; margin-bottom: 0.7em'><ul>
            <li>1) Please login to your Huggingface account</li>
            <li>2) Accept the SDXL 0.9 Research License Agreement <b><a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/tree/main'>here</a></b></li>
            <li>3) Create a new token at <b><a href='https://huggingface.co/settings/tokens'>here</a></b></li>
            <li>4) Set the Huggingface access token from the XL Demo menu in the Settings tab.</li>
            <li>5) Apply settings</li>
            <li>6) Restart the Web UI (not Reload UI)</li>
            </ul></div>""")

            return ui_component

    else:
        with gr.Blocks(analytics_enabled=False) as ui_component:

            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                xldemo_txt2img_prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3,
                                                                   placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])
                                xldemo_txt2img_dummy_component = gr.Label(
                                    visible=False)

                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                xldemo_txt2img_negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False,
                                                                            lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])

                with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
                    with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                        xldemo_txt2img_submit = gr.Button(
                            'Generate', elem_id=f"{id_part}_generate", variant='primary')

            with gr.Row():
                with gr.Column():
                    xldemo_txt2img_steps = gr.Slider(minimum=1, maximum=150, step=1,
                                                     elem_id=f"{id_part}_steps", label="Sampling steps", value=20)

                    with gr.Row():
                        with gr.Column(elem_id="xldemo_txt2img_column_size", scale=4):
                            xldemo_txt2img_width = gr.Slider(
                                minimum=64, maximum=2048, step=8, label="Width", value=1024, elem_id="xldemo_txt2img_width")
                            xldemo_txt2img_height = gr.Slider(
                                minimum=64, maximum=2048, step=8, label="Height", value=1024, elem_id="xldemo_txt2img_height")

                        with gr.Column(elem_id="xldemo_txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                            xldemo_txt2img_res_switch_btn = ToolButton(
                                value=switch_values_symbol, elem_id="xldemo_txt2img_res_switch_btn", label="Switch dims")

                        with gr.Column(elem_id="txt2img_column_batch"):
                            xldemo_txt2img_batch_size = gr.Slider(
                                minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{id_part}_batch_size")

                    xldemo_txt2img_cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                         label='CFG Scale', value=7.0, elem_id=f"{id_part}_cfg_scale")

                    xldemo_txt2img_seed, xldemo_txt2img_reuse_seed = create_seed_inputs(id_part)

                    xldemo_txt2img_enable_refiner = gr.Checkbox(
                            visible=XLDEMO_LOAD_REFINER_ON_STARTUP,
                            label='Refiner', value=False, elem_id=f"{id_part}_enable_refiner")

                    with gr.Row(visible=False, elem_id="xldemo_txt2img_refiner_group", variant="compact") as xldemo_txt2img_refiner_group:
                        xldemo_txt2img_refiner_strength = gr.Slider(
                            interactive=XLDEMO_LOAD_REFINER_ON_STARTUP,
                            label="Refiner Strength", minimum=0, maximum=1.0, value=0.3, step=0.1, elem_id=f"{id_part}_refiner_strength")

                with gr.Column():
                    xldemo_txt2img_gallery, xldemo_txt2img_generation_info, xldemo_txt2img_html_info, xldemo_txt2img_html_log = create_output_panel(
                        id_part, opts.outdir_txt2img_samples)

                connect_reuse_seed(xldemo_txt2img_seed, xldemo_txt2img_reuse_seed,
                                   xldemo_txt2img_generation_info, xldemo_txt2img_dummy_component, is_subseed=False)

                xldemo_txt2img_res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('xldemo_txt2img')}", inputs=None, outputs=None, show_progress=False)

                xldemo_txt2img_enable_refiner.change(
                    fn=lambda x: gr_show(x),
                    inputs=[xldemo_txt2img_enable_refiner],
                    outputs=[xldemo_txt2img_refiner_group],
                    show_progress = False,
                )

            # do_xldemo_txt2img_infer(prompt, negative, scale, samples=1, steps=20, refiner_strength=0.3)
            xldemo_txt2img_submit.click(fn=do_xldemo_txt2img_infer, inputs=[
                xldemo_txt2img_prompt,
                xldemo_txt2img_negative_prompt,
                xldemo_txt2img_width,
                xldemo_txt2img_height,
                xldemo_txt2img_cfg_scale,
                xldemo_txt2img_seed,
                xldemo_txt2img_batch_size,
                xldemo_txt2img_steps,
                xldemo_txt2img_enable_refiner,
                xldemo_txt2img_refiner_strength
            ], outputs=[
                xldemo_txt2img_gallery,
                xldemo_txt2img_generation_info,
                xldemo_txt2img_html_info,
                xldemo_txt2img_html_log,
            ], api_name="do_xldemo_txt2img_infer")

            return ui_component
