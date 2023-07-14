import gradio as gr

from modules.shared import opts
from modules.ui_components import ToolButton
from modules import sd_models 

from xldemo_txt2img import XLDEMO_HUGGINGFACE_ACCESS_TOKEN, XLDEMO_LOAD_REFINER_ON_STARTUP
from xldemo_txt2img import XLDEMO_SCHEDULER_CHOICES
from xldemo_txt2img import do_xldemo_txt2img_infer, do_xldemo_txt2img_refine, can_infer, can_refine
from xldemo_txt2img_ui_common import create_seed_inputs, create_output_panel, connect_reuse_seed, gr_show


switch_values_symbol = '\U000021C5'  # ⇅


def make_ui():
    id_part = 'xldemo_txt2img'

    if XLDEMO_HUGGINGFACE_ACCESS_TOKEN is None or XLDEMO_HUGGINGFACE_ACCESS_TOKEN.strip() == '':
        with gr.Blocks(analytics_enabled=False) as ui_component:
            gr.HTML(value="""<div style='font-size: 1.4em; margin-bottom: 0.7em'><ul>
            <li>*** It needs to have a GPU to run ***</li>
            <li>1) Please login to your Huggingface account</li>
            <li>2) Accept the SDXL 0.9 Research License Agreement <b><a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/tree/main'>here</a></b></li>
            <li>3) Accept the SDXL 1.0 Research License Agreement <b><a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main'>here</a></b></li>
            <li>4) Create a new token at <b><a href='https://huggingface.co/settings/tokens'>here</a></b></li>
            <li>5) Set the Huggingface access token from the XL Demo menu in the Settings tab.</li>
            <li>6) Set the model to be SDXL 0.9 (fp16) or SDXL 1.0 (fp16) if you did not downloaded any weights before.</li>
            <li>7) Apply settings</li>
            <li>8) Restart the Web UI (not Reload UI)</li>
            </ul></div>""")

            return ui_component

    else:
        with gr.Blocks(analytics_enabled=False) as ui_component:

            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                with gr.Column(scale=14, elem_id=f"{id_part}_prompt_container"):
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

                with gr.Column(scale=1, elem_id=f"{id_part}_memory_column"):
                        xldemo_txt2img_unload_sd_model = gr.Button(
                            'Unload SD checkpoint to free VRAM', elem_id=f"{id_part}_unload_sd_model")

                with gr.Column(scale=4, elem_id=f"{id_part}_actions_column"):
                    with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                        xldemo_txt2img_submit = gr.Button(
                            'Generate', elem_id=f"{id_part}_generate", variant='primary', interactive=can_infer())
                        
                    with gr.Row(elem_id=f"{id_part}_refine_box", elem_classes="refine-box"):
                        xldemo_txt2img_refine = gr.Button(                            
                            'Refine', interactive=False, elem_id=f"{id_part}_refine", variant='primary')

            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        xldemo_txt2img_sampler = gr.Dropdown(label='Sampling method', elem_id=f"{id_part}_sampling", choices=XLDEMO_SCHEDULER_CHOICES, value=XLDEMO_SCHEDULER_CHOICES[0])

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
                        with gr.Column():
                            xldemo_txt2img_image_to_refine = gr.Image(label="Image", type='pil')

                        with gr.Column():
                            xldemo_txt2img_refiner_steps = gr.Slider(minimum=1, maximum=150, step=1,
                                                            elem_id=f"{id_part}_refiner_steps", label="Refiner steps", value=20)

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

                xldemo_txt2img_enable_refiner.change(
                    fn=lambda x: gr.update(interactive=x and can_refine()),
                    inputs=[xldemo_txt2img_enable_refiner],
                    outputs=[xldemo_txt2img_refine],
                    show_progress = False,
                )

            with gr.Row():
                gr.HTML(value="<p style='font-size: 1.0em; margin-bottom: 0.7em'>Watch 📺 <b><a href=\"https://youtu.be/iF4w7gFDaYM\">video</a></b> for detailed explanation 🔍 ☕️ Please consider supporting me in Patreon <b><a href=\"https://www.patreon.com/lifeisboringsoprogramming\">here</a></b> 🍻</p>")


            xldemo_txt2img_submit.click(fn=do_xldemo_txt2img_infer, inputs=[
                xldemo_txt2img_prompt,
                xldemo_txt2img_negative_prompt,
                xldemo_txt2img_width,
                xldemo_txt2img_height,
                xldemo_txt2img_cfg_scale,
                xldemo_txt2img_seed,
                xldemo_txt2img_batch_size,
                xldemo_txt2img_sampler,
                xldemo_txt2img_steps
            ], outputs=[
                xldemo_txt2img_gallery,
                xldemo_txt2img_generation_info,
                xldemo_txt2img_html_info,
                xldemo_txt2img_html_log,
            ], api_name="do_xldemo_txt2img_infer")

            
            xldemo_txt2img_refine.click(fn=do_xldemo_txt2img_refine, inputs=[
                xldemo_txt2img_prompt,
                xldemo_txt2img_negative_prompt,
                xldemo_txt2img_seed,
                xldemo_txt2img_sampler,
                xldemo_txt2img_refiner_steps,
                xldemo_txt2img_enable_refiner,
                xldemo_txt2img_image_to_refine,
                xldemo_txt2img_refiner_strength
            ], outputs=[
                xldemo_txt2img_gallery,
                xldemo_txt2img_generation_info,
                xldemo_txt2img_html_info,
                xldemo_txt2img_html_log,
            ], api_name="do_xldemo_txt2img_refine")

            xldemo_txt2img_unload_sd_model.click(
                fn=sd_models.unload_model_weights,
                inputs=[],
                outputs=[]
            )

            return ui_component
