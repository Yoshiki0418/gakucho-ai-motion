"""最終調整: マテリアル未割当(WorldGridMaterial)のプリミティブを除去し、目の透明シェルを不可視化する。"""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from strip_textures import read_glb, write_glb
from shrink_textures import prune_unused

def main(src, dst):
    j, b = read_glb(src)
    mats = j['materials']
    removed = []
    for m in j['meshes']:
        keep = []
        for p in m['primitives']:
            name = mats[p['material']]['name'] if 'material' in p else ''
            if name == 'WorldGridMaterial':
                removed.append(f"{m['name']}:{name}")
                continue
            keep.append(p)
        m['primitives'] = keep
    for m in mats:
        if 'EyeShell' in m.get('name', ''):
            m.setdefault('pbrMetallicRoughness', {})['baseColorFactor'] = [0, 0, 0, 0]
            m['alphaMode'] = 'BLEND'
            removed.append(f"eye shell hidden: {m['name']}")
    prune_unused(j)
    write_glb(dst, j, b)
    print(f"{dst}: removed {removed}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
