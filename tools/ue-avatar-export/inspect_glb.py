"""glb のノード/スキン/メッシュ/マテリアル構成を要約する(標準ライブラリのみ)。"""
import json, struct, sys, collections

def load(path):
    with open(path, 'rb') as f:
        magic, ver, length = struct.unpack('<4sII', f.read(12))
        assert magic == b'glTF', magic
        clen, ctype = struct.unpack('<II', f.read(8))
        j = json.loads(f.read(clen))
        rest = length - 20 - clen
    return j, rest

def main(path):
    j, binlen = load(path)
    nodes = j.get('nodes', [])
    print(f"== {path}")
    print(f"  bin bytes={binlen:,} nodes={len(nodes)} meshes={len(j.get('meshes',[]))} skins={len(j.get('skins',[]))} "
          f"materials={len(j.get('materials',[]))} textures={len(j.get('textures',[]))} images={len(j.get('images',[]))} "
          f"extensionsUsed={j.get('extensionsUsed')}")
    acc = j.get('accessors', [])
    for mi, m in enumerate(j.get('meshes', [])):
        tot = 0; tri = 0; targets = 0
        for p in m['primitives']:
            tot += acc[p['attributes']['POSITION']]['count']
            if 'indices' in p: tri += acc[p['indices']]['count'] // 3
            targets = max(targets, len(p.get('targets', [])))
            attrs = sorted(p['attributes'].keys())
        print(f"  mesh[{mi}] '{m.get('name')}' prims={len(m['primitives'])} verts={tot:,} tris={tri:,} morph={targets} attrs={attrs}")
    for si, s in enumerate(j.get('skins', [])):
        names = [nodes[k].get('name') for k in s['joints']]
        print(f"  skin[{si}] '{s.get('name')}' joints={len(names)} skeleton={s.get('skeleton')} first={names[:12]}")
    # 各スキンドメッシュ
    for ni, n in enumerate(nodes):
        if 'skin' in n:
            print(f"  node[{ni}] '{n.get('name')}' mesh={n.get('mesh')} skin={n.get('skin')} t={n.get('translation')} r={n.get('rotation')} s={n.get('scale')}")
    for mi, m in enumerate(j.get('materials', [])):
        pbr = m.get('pbrMetallicRoughness', {})
        print(f"  mat[{mi}] '{m.get('name')}' baseTex={'baseColorTexture' in pbr} baseColor={pbr.get('baseColorFactor')} "
              f"normal={'normalTexture' in m} alphaMode={m.get('alphaMode')} doubleSided={m.get('doubleSided')}")
    imgs = j.get('images', [])
    bvs = j.get('bufferViews', [])
    for ii, im in enumerate(imgs[:40]):
        bl = bvs[im['bufferView']]['byteLength'] if 'bufferView' in im else -1
        print(f"  image[{ii}] '{im.get('name')}' {im.get('mimeType')} bytes={bl:,}")

for p in sys.argv[1:]:
    main(p)
