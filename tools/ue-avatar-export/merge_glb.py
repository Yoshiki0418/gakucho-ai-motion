"""複数の glb を 1 つに結合する(各 glb のシーンルートノードを 1 シーンにまとめる)。
usage: merge_glb.py out.glb in1.glb in2.glb ...
"""
import json, struct, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from strip_textures import read_glb, write_glb

LIST_KEYS = ['accessors', 'bufferViews', 'images', 'materials', 'meshes', 'nodes', 'samplers', 'skins', 'textures', 'cameras']

def main(out, inputs):
    merged = {'asset': {'version': '2.0', 'generator': 'merge_glb'}, 'scene': 0, 'scenes': [{'nodes': []}]}
    for k in LIST_KEYS: merged[k] = []
    ext_used = set(); ext_req = set()
    blob = bytearray()
    for path in inputs:
        j, b = read_glb(path)
        off = {k: len(merged[k]) for k in LIST_KEYS}
        while len(blob) % 4: blob.append(0)
        bin_off = len(blob); blob += b
        ext_used |= set(j.get('extensionsUsed', [])); ext_req |= set(j.get('extensionsRequired', []))
        for bv in j.get('bufferViews', []):
            nb = dict(bv); nb['buffer'] = 0; nb['byteOffset'] = bv.get('byteOffset', 0) + bin_off
            merged['bufferViews'].append(nb)
        for a in j.get('accessors', []):
            na = json.loads(json.dumps(a))
            if 'bufferView' in na: na['bufferView'] += off['bufferViews']
            sp = na.get('sparse')
            if sp:
                sp['indices']['bufferView'] += off['bufferViews']; sp['values']['bufferView'] += off['bufferViews']
            merged['accessors'].append(na)
        for im in j.get('images', []):
            ni = dict(im); ni['bufferView'] += off['bufferViews']; merged['images'].append(ni)
        for sm in j.get('samplers', []): merged['samplers'].append(dict(sm))
        for t in j.get('textures', []):
            nt = dict(t)
            if 'source' in nt: nt['source'] += off['images']
            if 'sampler' in nt: nt['sampler'] += off['samplers']
            merged['textures'].append(nt)
        def fix_tex(d):
            if isinstance(d, dict):
                if 'index' in d and 'texCoord' in d or ('index' in d and len(d) <= 3 and 'scale' in d or 'strength' in d):
                    d['index'] += off['textures']
                elif 'index' in d and set(d) <= {'index', 'texCoord', 'scale', 'strength', 'extensions'}:
                    d['index'] += off['textures']
                else:
                    for v in d.values(): fix_tex(v)
            elif isinstance(d, list):
                for v in d: fix_tex(v)
        for m in j.get('materials', []):
            nm = json.loads(json.dumps(m)); fix_tex(nm); merged['materials'].append(nm)
        for me in j.get('meshes', []):
            nm = json.loads(json.dumps(me))
            for p in nm['primitives']:
                p['attributes'] = {k: v + off['accessors'] for k, v in p['attributes'].items()}
                if 'indices' in p: p['indices'] += off['accessors']
                if 'material' in p: p['material'] += off['materials']
                if 'targets' in p: p['targets'] = [{k: v + off['accessors'] for k, v in t.items()} for t in p['targets']]
            merged['meshes'].append(nm)
        for sk in j.get('skins', []):
            ns = json.loads(json.dumps(sk))
            ns['joints'] = [x + off['nodes'] for x in ns['joints']]
            if 'skeleton' in ns: ns['skeleton'] += off['nodes']
            if 'inverseBindMatrices' in ns: ns['inverseBindMatrices'] += off['accessors']
            merged['skins'].append(ns)
        for n in j.get('nodes', []):
            nn = json.loads(json.dumps(n))
            if 'children' in nn: nn['children'] = [x + off['nodes'] for x in nn['children']]
            if 'mesh' in nn: nn['mesh'] += off['meshes']
            if 'skin' in nn: nn['skin'] += off['skins']
            if 'camera' in nn: nn['camera'] += off['cameras']
            merged['nodes'].append(nn)
        for c in j.get('cameras', []): merged['cameras'].append(dict(c))
        scene = j['scenes'][j.get('scene', 0)]
        merged['scenes'][0]['nodes'] += [x + off['nodes'] for x in scene.get('nodes', [])]
        print(f"  + {path}: nodes={len(j.get('nodes', []))} skins={len(j.get('skins', []))} bin={len(b):,}")
    for k in LIST_KEYS:
        if not merged[k]: del merged[k]
    if ext_used: merged['extensionsUsed'] = sorted(ext_used)
    if ext_req: merged['extensionsRequired'] = sorted(ext_req)
    merged['buffers'] = [{'byteLength': len(blob)}]
    write_glb(out, merged, bytes(blob))
    print(f"-> {out}: nodes={len(merged.get('nodes', []))} skins={len(merged.get('skins', []))} bin={len(blob):,}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
