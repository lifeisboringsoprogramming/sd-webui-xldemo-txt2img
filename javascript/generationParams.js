// attaches listeners to the xldemo_txt2img and img2img galleries to update displayed generation param text when the image changes

let xldemo_txt2img_gallery, xldemo_txt2img_modal = undefined;
onAfterUiUpdate(function() {
    if (!xldemo_txt2img_gallery) {
        xldemo_txt2img_gallery = attachGalleryListeners("xldemo_txt2img");
    }
    if (!xldemo_txt2img_modal) {
        xldemo_txt2img_modal = gradioApp().getElementById('lightboxModal');
        xldemo_txt2img_modalObserver.observe(xldemo_txt2img_modal, {attributes: true, attributeFilter: ['style']});
    }
});

let xldemo_txt2img_modalObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutationRecord) {
        let selectedTab = gradioApp().querySelector('#tabs div button.selected')?.innerText;
        if (mutationRecord.target.style.display === 'none' && (selectedTab === 'xldemo_txt2img')) {
            gradioApp().getElementById(selectedTab + "_generation_info_button")?.click();
        }
    });
});

function attachGalleryListeners(tab_name) {
    var gallery = gradioApp().querySelector('#' + tab_name + '_gallery');
    gallery?.addEventListener('click', () => gradioApp().getElementById(tab_name + "_generation_info_button").click());
    gallery?.addEventListener('keydown', (e) => {
        if (e.keyCode == 37 || e.keyCode == 39) { // left or right arrow
            gradioApp().getElementById(tab_name + "_generation_info_button").click();
        }
    });
    return gallery;
}
