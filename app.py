import io, zipfile, base64
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
from openai import OpenAI

st.set_page_config(page_title="UV Artwork Studio", page_icon="🖨️", layout="wide")
st.title("UV Artwork Studio")
st.caption("Customer demo → AI artwork → automatic crop → optional silhouette")

FIXED_PROMPT = """Convert the provided 3D product/sign photograph into a clean, flat 2D front-facing logo artwork for UV printing.

Extract and recreate the exact logo, typography, colors, icons, proportions, and wording from the reference image. Remove all 3D perspective, depth, bevels, shadows, reflections, highlights, lighting, background, table, plants, and environmental elements.

The final artwork must look like a straight-on high-resolution 2D print file, as if the logo was designed digitally from scratch.

Keep the white area surrounding the entire logo/design exactly as part of the artwork, including the white outline/border around the shield, text, and lower rectangular section. This white border should follow the exact outer contour of the logo.

Important: Anything outside the outer white outline must be fully transparent PNG, not white.

Do not add a rectangular white background. Do not crop or remove any part of the logo. Preserve the complete bottom section and its straight horizontal bottom edge.

Output a PNG with a genuine alpha-transparent background. The white logo border itself must remain opaque white. Do not render a checkerboard pattern."""

def crop_transparent(im, padding=0):
    """Crop only fully/mostly transparent exterior pixels. Opaque white outline is retained."""
    im = im.convert("RGBA")
    alpha = im.getchannel("A")
    bbox = alpha.point(lambda p: 255 if p > 5 else 0).getbbox()
    if not bbox:
        return im
    l,t,r,b = bbox
    l=max(0,l-padding); t=max(0,t-padding); r=min(im.width,r+padding); b=min(im.height,b+padding)
    return im.crop((l,t,r,b))

def make_silhouette(im, threshold=128, color="#000000", invert=False):
    """Use alpha as the complete silhouette boundary; white artwork/outline stays included."""
    rgba=im.convert("RGBA")
    alpha=rgba.getchannel("A")
    # AI artwork should have transparency outside the logo. The whole visible logo,
    # including white border and internal white areas, is treated as the silhouette.
    mask=alpha.point(lambda p: 255 if (p >= threshold) != invert else 0)
    layer=Image.new("RGBA", rgba.size, color)
    layer.putalpha(mask)
    return layer

def silhouette_svg(im, color="#000000", threshold=128):
    """Trace the silhouette alpha mask into a true vector SVG with holes preserved."""
    rgba = im.convert("RGBA")
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda p: 255 if p >= threshold else 0)
    px = mask.load()
    w, h = mask.size
    edges = {}
    def add(a, b):
        edges.setdefault(a, []).append(b)
    # Clockwise boundary edges around every filled pixel; shared edges cancel.
    for y in range(h):
        for x in range(w):
            if px[x, y] == 0:
                continue
            if y == 0 or px[x, y-1] == 0: add((x,y), (x+1,y))
            if x == w-1 or px[x+1,y] == 0: add((x+1,y), (x+1,y+1))
            if y == h-1 or px[x,y+1] == 0: add((x+1,y+1), (x,y+1))
            if x == 0 or px[x-1,y] == 0: add((x,y+1), (x,y))
    paths = []
    while edges:
        start = next(iter(edges))
        cur = start
        points = [cur]
        guard = 0
        while guard < (w*h*4 + 10):
            guard += 1
            opts = edges.get(cur)
            if not opts:
                break
            nxt = opts.pop()
            if not opts:
                del edges[cur]
            cur = nxt
            if cur == start:
                break
            points.append(cur)
        if len(points) >= 3 and cur == start:
            # Remove collinear intermediate points to keep SVG compact.
            simplified=[]
            for i,pt in enumerate(points):
                prev=points[i-1]; nxt=points[(i+1)%len(points)]
                if (pt[0]-prev[0])*(nxt[1]-pt[1]) != (pt[1]-prev[1])*(nxt[0]-pt[0]):
                    simplified.append(pt)
            if len(simplified) >= 3:
                d = f"M {simplified[0][0]} {simplified[0][1]} " + " ".join(f"L {x} {y}" for x,y in simplified[1:]) + " Z"
                paths.append(d)
    path_data = " ".join(paths)
    safe_color = color if color.startswith("#") and len(color) in (4,7) else "#000000"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}px" height="{h}px" '
            f'viewBox="0 0 {w} {h}"><path d="{path_data}" fill="{safe_color}" '
            'fill-rule="evenodd"/></svg>')

def png_bytes(im):
    b=io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()

with st.sidebar:
    st.header("AI settings")
    model = st.selectbox("AI model", ["gpt-image-2.5-sunburst", "gpt-image-2"], index=0)
    quality = st.selectbox("Quality", ["medium", "high"], index=0)
    size = st.selectbox("Output size", ["1536x1024", "1024x1024", "1024x1536"], index=0)
    api_key = st.text_input("OpenAI API key", type="password", help="Used for this session; not saved by the app.")
    st.caption("API usage is billed separately from ChatGPT.")
    st.divider()
    st.header("Crop settings")
    padding = st.number_input("Extra transparent margin (pixels)", min_value=0, max_value=500, value=0, step=5)
    st.caption("Cropping detects transparency only. White pixels and the white logo outline are never removed.")

st.subheader("1. Upload customer demo")
uploads = st.file_uploader("Choose one or more 3D demo images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
st.subheader("2. Silhouette options")
make_sil = st.checkbox("Also create silhouette files", value=True)
col1,col2,col3=st.columns(3)
with col1: sil_color=st.selectbox("Silhouette colour", ["Black", "White", "Charcoal", "Red", "Blue"], index=0)
with col2: sil_threshold=st.slider("Alpha threshold", 1, 255, 128)
with col3: invert=st.checkbox("Invert silhouette", value=False)
colors={"Black":"#000000","White":"#ffffff","Charcoal":"#1a1a1a","Red":"#c0392b","Blue":"#2057c7"}

if uploads:
    st.success(f"{len(uploads)} demo image(s) selected")
    if st.button("3. Generate artwork and crop automatically", type="primary", disabled=not api_key):
        client=OpenAI(api_key=api_key)
        results=[]; errors=[]
        bar=st.progress(0); status=st.empty()
        for i,file in enumerate(uploads):
            try:
                raw=file.getvalue()
                im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                buf=io.BytesIO(); im.save(buf,format="PNG"); buf.seek(0)
                status.write(f"AI processing {i+1}/{len(uploads)}: {file.name}")
                response=client.images.edit(
                    model=model,
                    image=(Path(file.name).stem+".png",buf.getvalue(),"image/png"),
                    prompt=FIXED_PROMPT,
                    background="transparent", output_format="png", quality=quality, size=size,
                )
                generated=Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))).convert("RGBA")
                cropped=crop_transparent(generated, int(padding))
                stem=Path(file.name).stem
                item={"name":stem,"art":png_bytes(cropped),"sil":None,"svg":None,"dims":cropped.size}
                if make_sil:
                    sil=make_silhouette(cropped, int(sil_threshold), colors[sil_color], invert)
                    item["sil"]=png_bytes(sil)
                    item["svg"]=silhouette_svg(sil, colors[sil_color], int(sil_threshold)).encode("utf-8")
                results.append(item)
            except Exception as e:
                errors.append((file.name,str(e)))
            bar.progress((i+1)/len(uploads))
        st.session_state["results"]=results
        st.session_state["errors"]=errors
        status.write(f"Finished: {len(results)} successful, {len(errors)} failed.")

if st.session_state.get("results"):
    st.subheader("4. Review and download")
    results=st.session_state["results"]
    for idx,item in enumerate(results):
        with st.expander(f"{item['name']} — {item['dims'][0]} × {item['dims'][1]} px", expanded=True):
            cols=st.columns(2 if item["sil"] else 1)
            with cols[0]:
                st.markdown("**Cropped full-colour artwork**")
                st.image(item["art"], use_container_width=True)
                st.download_button("Download cropped artwork PNG",item["art"],file_name=item["name"]+"_UV_cropped.png",mime="image/png",key=f"art{idx}")
            if item["sil"]:
                with cols[1]:
                    st.markdown("**Silhouette (includes complete white outline)**")
                    st.image(item["sil"], use_container_width=True)
                    st.download_button("Download silhouette PNG",item["sil"],file_name=item["name"]+"_silhouette.png",mime="image/png",key=f"sil{idx}")
                    st.download_button("Download silhouette SVG (vector)",item["svg"],file_name=item["name"]+"_silhouette.svg",mime="image/svg+xml",key=f"svg{idx}")
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for item in results:
            z.writestr(item["name"]+"_UV_cropped.png",item["art"])
            if item["sil"]:
                z.writestr(item["name"]+"_silhouette.png",item["sil"])
                z.writestr(item["name"]+"_silhouette.svg",item["svg"])
    st.download_button("5. Download all files (ZIP)",archive.getvalue(),file_name="UV_production_files.zip",mime="application/zip",type="primary")
if st.session_state.get("errors"):
    st.subheader("Images needing attention")
    for name,error in st.session_state["errors"]: st.error(f"{name}: {error}")
