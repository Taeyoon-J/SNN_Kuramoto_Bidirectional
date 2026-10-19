"""Exercise the local, self-contained diagram and export browser-rendered PNG/PDF."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import json

p=Path(__file__).resolve().parents[2]/'collaborative_testing/generated/architecture_figures/native64_interactive'
with sync_playwright() as a:
    b=a.chromium.launch(headless=True)
    page=b.new_page(viewport={'width':1900,'height':1350},device_scale_factor=1.5)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((p/'native64_interactive.html').as_uri())
    def slide(value):
        page.locator('#time').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',str(value))
    page.locator('.canvas svg').screenshot(path=str(p/'native64_visual_architecture.png'))
    page.locator('[data-stage="projection"]').click()
    assert 'sensory projection' in page.locator('#detail-title').inner_text()
    assert page.locator('#zoom svg').count()==1
    slide(1)
    assert page.locator('#time-label').inner_text()=='129'
    assert page.locator('#phase-2-0').get_attribute('href')==page.evaluate('frames[1][0]')
    page.locator('#feedback').uncheck()
    assert page.locator('#disabled-paths').evaluate('(e)=>getComputedStyle(e).display')=='none'
    page.locator('#feedback').check()
    slide(3)
    page.locator('#play').click()
    page.wait_for_timeout(950)
    assert page.locator('#time-label').inner_text()!='1024'
    page.locator('#play').click()
    slide(3)
    with page.expect_download() as download:
        page.locator('#save').click()
    assert download.value.suggested_filename.endswith('.svg')
    page.screenshot(path=str(p/'interactive_preview.png'),full_page=True)
    page.pdf(path=str(p/'native64_visual_architecture.pdf'),print_background=True,prefer_css_page_size=True)
    assert not errors,errors
    (p/'browser_verification.json').write_text(json.dumps(dict(
        stage_selection=True,image_enlargement=True,phase_slider=True,animation=True,
        disabled_path_toggle=True,svg_download=True,javascript_errors=errors,
        png_and_pdf_exported=True),indent=2),encoding='utf-8')
    print('PASS: click, zoom, slider, animation, feedback toggle, SVG download; PNG/PDF exported; no JS errors.')
    b.close()
