"""glb 内の画像を上限解像度へ縮小し(PIL)、bufferView を再パックする。
usage: shrink_textures.py in.glb out.glb [max_size=1024] [jpeg_quality=0(=PNG維持)]
"""
import io, json, struct, sys
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from strip_textures import read_glb, write_glb

def prune_unused(j):
    """プリミティブから参照されないマテリアル → テクスチャ → 画像を落とす(全 LOD 分のスロットが書き出されるため)。"""
    used_mat = {p['material'] for m in j.get('meshes', []) for p in m['primitives'] if 'material' in p}
    mats = j.get('materials', []); keep = [i for i in range(len(mats)) if i in used_mat]; remap = {o: n for n, o in enumerate(keep)}
    j['materials'] = [mats[i] for i in keep]
    for m in j.get('meshes', []):
        for p in m['primitives']:
            if 'material' in p: p['material'] = remap[p['material']]
    def tex_refs(d, acc):
        if isinstance(d, dict):
            if 'index' in d and isinstance(d['index'], int) and set(d) <= {'index', 'texCoord', 'scale', 'strength', 'extensions'}:
                acc.append(d)
            else:
                for v in d.values(): tex_refs(v, acc)
        elif isinstance(d, list):
            for v in d: tex_refs(v, acc)
    refs = []; tex_refs(j['materials'], refs)
    texs = j.get('textures', []); keep_t = sorted({r['index'] for r in refs}); remap_t = {o: n for n, o in enumerate(keep_t)}
    for r in refs: r['index'] = remap_t[r['index']]
    j['textures'] = [texs[i] for i in keep_t]
    imgs = j.get('images', []); keep_i = sorted({t['source'] for t in j['textures'] if 'source' in t}); remap_i = {o: n for n, o in enumerate(keep_i)}
    for t in j['textures']:
        if 'source' in t: t['source'] = remap_i[t['source']]
    j['images'] = [imgs[i] for i in keep_i]
    print(f"  prune: materials {len(mats)}->{len(j['materials'])} textures {len(texs)}->{len(j['textures'])} images {len(imgs)}->{len(j['images'])}")

def main(src, dst, max_size=1024, jpeg_quality=0):
    j, b = read_glb(src)
    prune_unused(j)
    bvs = j['bufferViews']
    new_blobs = {}  # image index -> (bytes, mime)
    for ii, im in enumerate(j.get('images', [])):
        bv = bvs[im['bufferView']]
        off = bv.get('byteOffset', 0); raw = b[off:off + bv['byteLength']]
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        name = im.get('name', '')
        # 法線・ラフネス/メタリック・オクルージョン・スペキュラは JPEG ノイズが目立つので PNG のまま
        is_normal = any(k in name.lower() for k in ('normal', 'metallicroughness', 'occlusion', 'specular', 'roughness'))
        if max(w, h) > max_size:
            s = max_size / max(w, h)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        out = io.BytesIO()
        has_alpha = img.mode in ('RGBA', 'LA') and img.getextrema()[-1][0] < 255
        if jpeg_quality and not is_normal and not has_alpha:
            img.convert('RGB').save(out, 'JPEG', quality=jpeg_quality, optimize=True)
            mime = 'image/jpeg'
        else:
            img.save(out, 'PNG', optimize=True)
            mime = 'image/png'
        new_blobs[ii] = (out.getvalue(), mime)
        print(f"  image[{ii}] {name}: {w}x{h} {len(raw):,}B -> {img.size[0]}x{img.size[1]} {len(out.getvalue()):,}B {mime}")
    # 再パック
    image_bv = {im['bufferView']: ii for ii, im in enumerate(j.get('images', []))}
    used = set(image_bv)
    for a in j.get('accessors', []):
        if 'bufferView' in a: used.add(a['bufferView'])
    out = bytearray(); remap = {}; new_bvs = []
    for i, bv in enumerate(bvs):
        if i not in used: continue
        while len(out) % 4: out.append(0)
        if i in image_bv:
            data, mime = new_blobs[image_bv[i]]
            j['images'][image_bv[i]]['mimeType'] = mime
        else:
            data = b[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]
        nb = {k: v for k, v in bv.items() if k not in ('byteOffset', 'byteLength')}
        nb.update(buffer=0, byteOffset=len(out), byteLength=len(data))
        out += data; remap[i] = len(new_bvs); new_bvs.append(nb)
    for a in j.get('accessors', []):
        if 'bufferView' in a: a['bufferView'] = remap[a['bufferView']]
    for im in j.get('images', []):
        im['bufferView'] = remap[im['bufferView']]
    j['bufferViews'] = new_bvs; j['buffers'] = [{'byteLength': len(out)}]
    write_glb(dst, j, bytes(out))
    print(f"{src} -> {dst}: bin {len(b):,} -> {len(out):,} bytes")

if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 1024, int(a[4]) if len(a) > 4 else 0)
