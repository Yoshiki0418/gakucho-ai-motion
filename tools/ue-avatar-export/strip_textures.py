"""glb からテクスチャ/画像を取り除き、未参照 bufferView を捨てて再パックする。"""
import json, struct, sys

def read_glb(path):
    with open(path, 'rb') as f:
        magic, ver, length = struct.unpack('<4sII', f.read(12))
        clen, ctype = struct.unpack('<II', f.read(8))
        j = json.loads(f.read(clen))
        blen, btype = struct.unpack('<II', f.read(8))
        b = f.read(blen)
    return j, b

def write_glb(path, j, b):
    js = json.dumps(j, separators=(',', ':')).encode()
    js += b' ' * ((4 - len(js) % 4) % 4)
    b += b'\0' * ((4 - len(b) % 4) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(b)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack('<II', len(b), 0x004E4942)); f.write(b)

def main(src, dst, base_color=None):
    j, b = read_glb(src)
    for m in j.get('materials', []):
        pbr = m.setdefault('pbrMetallicRoughness', {})
        for k in ('baseColorTexture', 'metallicRoughnessTexture'):
            pbr.pop(k, None)
        for k in ('normalTexture', 'occlusionTexture', 'emissiveTexture'):
            m.pop(k, None)
        m.pop('extensions', None)  # KHR_materials_specular 等のテクスチャ参照も落とす
        if base_color:
            pbr['baseColorFactor'] = base_color
        pbr.setdefault('metallicFactor', 0.0)
        pbr.setdefault('roughnessFactor', 0.8)
    j.pop('textures', None); j.pop('images', None); j.pop('samplers', None)
    j.pop('extensionsUsed', None); j.pop('extensionsRequired', None)
    # 参照されている bufferView だけ残す
    used = set()
    for a in j.get('accessors', []):
        if 'bufferView' in a: used.add(a['bufferView'])
        sp = a.get('sparse')
        if sp:
            used.add(sp['indices']['bufferView']); used.add(sp['values']['bufferView'])
    old_bvs = j['bufferViews']; new_bvs = []; remap = {}; out = bytearray()
    for i, bv in enumerate(old_bvs):
        if i not in used: continue
        off = bv.get('byteOffset', 0); ln = bv['byteLength']
        while len(out) % 4: out.append(0)
        nb = dict(bv); nb['byteOffset'] = len(out); nb['buffer'] = 0
        out += b[off:off + ln]
        remap[i] = len(new_bvs); new_bvs.append(nb)
    for a in j.get('accessors', []):
        if 'bufferView' in a: a['bufferView'] = remap[a['bufferView']]
        sp = a.get('sparse')
        if sp:
            sp['indices']['bufferView'] = remap[sp['indices']['bufferView']]
            sp['values']['bufferView'] = remap[sp['values']['bufferView']]
    j['bufferViews'] = new_bvs
    j['buffers'] = [{'byteLength': len(out)}]
    write_glb(dst, j, bytes(out))
    print(f"{src} -> {dst}: bin {len(b):,} -> {len(out):,} bytes")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
