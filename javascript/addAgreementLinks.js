
onAfterUiUpdate(function () {
    const divId = "setting_xldemo_txt2img_agreement_links";
    let xlDemoTxt2ImgAgreementLinksDiv = document.getElementById(divId);
    if (xlDemoTxt2ImgAgreementLinksDiv == null) {

        xlDemoTxt2ImgAgreementLinksDiv = document.createElement("div");
        xlDemoTxt2ImgAgreementLinksDiv.id = divId;
        const lines = [
            "<p style='margin-top: 5px;'>Accept the SDXL 0.9 Research License Agreement <b><a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/tree/main'>here</a></b></p>",
            "<p>Accept the SDXL 1.0 Research License Agreement <b><a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main'>here</a></b></p>"
        ];
        xlDemoTxt2ImgAgreementLinksDiv.innerHTML = lines.join("");
    
        const settingUiDiv = document.getElementById("setting_xldemo_txt2img_model");
    
        if (settingUiDiv && settingUiDiv.parentNode) {
            settingUiDiv.parentNode.insertBefore(xlDemoTxt2ImgAgreementLinksDiv, settingUiDiv);
        }
    }

});
