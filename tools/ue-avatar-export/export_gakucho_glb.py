"""
UE エディタ内で実行し、学長 MetaHuman を glb へ書き出す。
起動: UnrealEditor-Cmd MyProject.uproject -ExecutePythonScript=<this file>
"""
import os
import traceback

import unreal

OUT_DIR = os.environ.get("GAKUCHO_GLB_OUT", os.path.expanduser("~/gakucho_glb_export"))
os.makedirs(OUT_DIR, exist_ok=True)
STATUS = open(os.path.join(OUT_DIR, "status.txt"), "a")


def log(msg):
    line = f"[gakucho-export] {msg}"
    unreal.log(line)
    STATUS.write(line + "\n")
    STATUS.flush()


def set_prop(obj, name, value):
    try:
        obj.set_editor_property(name, value)
    except Exception as e:  # noqa: BLE001
        log(f"  (skip option {name}: {e})")


def make_options(lod=0, bake=True, morph=False, bake_size=2048):
    o = unreal.GLTFExportOptions()
    set_prop(o, "export_uniform_scale", 0.01)
    set_prop(o, "export_preview_mesh", False)
    set_prop(o, "bake_material_inputs",
             unreal.GLTFMaterialBakeMode.SIMPLE if bake else unreal.GLTFMaterialBakeMode.DISABLED)
    try:
        size = unreal.GLTFMaterialBakeSize()
        size.set_editor_property("x", bake_size)
        size.set_editor_property("y", bake_size)
        set_prop(o, "default_material_bake_size", size)
    except Exception as e:  # noqa: BLE001
        log(f"  (skip bake size: {e})")
    set_prop(o, "default_level_of_detail", lod)
    set_prop(o, "export_vertex_skin_weights", True)
    set_prop(o, "export_vertex_colors", False)
    set_prop(o, "export_morph_targets", morph)
    set_prop(o, "export_animation_sequences", False)
    set_prop(o, "export_level_sequences", False)
    set_prop(o, "export_lights", False)
    set_prop(o, "export_cameras", False)
    set_prop(o, "export_hidden_in_game", True)
    set_prop(o, "use_mesh_quantization", False)
    set_prop(o, "texture_image_format", unreal.GLTFTextureImageFormat.PNG)
    return o


def do_export(obj, filename, opts, actors=None):
    path = os.path.join(OUT_DIR, filename)
    log(f"exporting {obj.get_name()} -> {path}")
    try:
        sel = unreal.Set(unreal.Actor)
        for a in actors or []:
            sel.add(a)
        res = unreal.GLTFExporter.export_to_gltf(obj, path, opts, sel)
        ok, msgs = (res if isinstance(res, tuple) else (res, None))
        if msgs is not None:
            for w in list(msgs.get_editor_property("warnings"))[:30]:
                log(f"  warn: {w}")
            for e in list(msgs.get_editor_property("errors"))[:30]:
                log(f"  error: {e}")
        size = os.path.getsize(path) if os.path.exists(path) else -1
        log(f"  result={ok} size={size}")
        return bool(ok)
    except Exception:  # noqa: BLE001
        log("  EXCEPTION:\n" + traceback.format_exc())
        return False


def describe_skm(skm):
    try:
        sub = unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem)
        lods = sub.get_lod_count(skm)
        verts = [sub.get_num_verts(skm, i) for i in range(lods)]
        log(f"  {skm.get_name()}: lods={lods} verts={verts}")
        return lods
    except Exception as e:  # noqa: BLE001
        log(f"  describe failed: {e}")
        return 1
    finally:
        try:
            mats = skm.get_editor_property("materials")
            log("  materials: " + ", ".join(
                (m.material_interface.get_name() if m.material_interface else "None") for m in mats))
            morphs = skm.get_all_morph_target_names()
            log(f"  morph targets: {len(morphs)}")
        except Exception as e:  # noqa: BLE001
            log(f"  (material/morph info failed: {e})")


def export_skm(asset_path, base_name, lods_to_try=(0,)):
    skm = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not skm:
        log(f"asset not found: {asset_path}")
        return
    lod_count = describe_skm(skm)
    for lod in lods_to_try:
        if lod >= lod_count:
            continue
        do_export(skm, f"{base_name}_lod{lod}.glb", make_options(lod=lod))


def export_actor(bp_path, base_name):
    cls = unreal.EditorAssetLibrary.load_blueprint_class(bp_path)
    if not cls:
        log(f"blueprint not found: {bp_path}")
        return
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actor_sub.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    if not actor:
        log(f"spawn failed: {bp_path}")
        return
    try:
        comps = actor.get_components_by_class(unreal.SkinnedMeshComponent)
        for c in comps:
            try:
                m = c.get_skinned_asset() if hasattr(c, "get_skinned_asset") else c.get_editor_property("skeletal_mesh")
                log(f"  component {c.get_name()} -> {m.get_name() if m else None}")
            except Exception as e:  # noqa: BLE001
                log(f"  component {c.get_name()} (mesh lookup failed: {e})")
        do_export(world, f"{base_name}.glb", make_options(lod=0), actors=[actor])
    finally:
        actor_sub.destroy_actor(actor)


def set_component_mesh(comp, skm):
    for name in ("set_skeletal_mesh_asset", "set_skeletal_mesh", "set_skinned_asset_and_update"):
        fn = getattr(comp, name, None)
        if fn:
            try:
                fn(skm)
                return True
            except Exception as e:  # noqa: BLE001
                log(f"  ({name} failed: {e})")
    try:
        comp.set_editor_property("skeletal_mesh_asset", skm)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  (set_editor_property skeletal_mesh_asset failed: {e})")
    return False


def make_export_safe(skm, lods):
    """GLTFExporter のスキンウェイト変換バグ(16bit ウェイト/可変影響数でヒープ破壊)を避けるため、
    書き出す LOD のビルド設定を 8bit ウェイト・影響数上限 8 に変更してリビルドする(保存はしない)。"""
    sub = unreal.SkeletalMeshEditorSubsystem
    try:
        lod_count = sub.get_lod_count(skm)
    except Exception as e:  # noqa: BLE001
        log(f"  (lod count failed: {e})")
        return
    for lod in lods:
        if lod >= lod_count:
            continue
        try:
            bs = sub.get_lod_build_settings(skm, lod)
            before = (bs.get_editor_property("use_high_precision_skin_weights"),
                      bs.get_editor_property("bone_influence_limit"))
            bs.set_editor_property("use_high_precision_skin_weights", False)
            bs.set_editor_property("bone_influence_limit", 8)
            sub.set_lod_build_settings(skm, lod, bs)
            log(f"  build settings LOD{lod}: highPrecisionWeights/influenceLimit {before} -> (False, 8)")
        except Exception as e:  # noqa: BLE001
            log(f"  (build settings LOD{lod} failed: {e})")


def export_mesh_set(asset_paths, base_name, lods=(0,), bake_size=1024):
    """複数の SkeletalMesh を SkeletalMeshActor として原点に置き、まとめて 1 つの glb に書き出す。"""
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = []
    try:
        for path in asset_paths:
            skm = unreal.EditorAssetLibrary.load_asset(path)
            if not skm:
                log(f"  asset not found: {path}")
                continue
            describe_skm(skm)
            make_export_safe(skm, lods)
            actor = actor_sub.spawn_actor_from_class(
                unreal.SkeletalMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
            actor.set_actor_label(skm.get_name())
            comp = actor.skeletal_mesh_component
            if not set_component_mesh(comp, skm):
                actor_sub.destroy_actor(actor)
                continue
            actors.append(actor)
        if not actors:
            log(f"  no actors for {base_name}")
            return
        for lod in lods:
            do_export(world, f"{base_name}_lod{lod}.glb",
                      make_options(lod=lod, bake_size=bake_size), actors=actors)
    finally:
        for a in actors:
            try:
                actor_sub.destroy_actor(a)
            except Exception:  # noqa: BLE001
                pass


def export_static_mesh(asset_path, base_name, bake_size=1024):
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    sm = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not sm:
        log(f"  asset not found: {asset_path}")
        return
    actor = actor_sub.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    try:
        actor.set_actor_label(sm.get_name())
        actor.static_mesh_component.set_static_mesh(sm)
        do_export(world, f"{base_name}.glb", make_options(lod=0, bake_size=bake_size), actors=[actor])
    finally:
        actor_sub.destroy_actor(actor)


V3_SET = [
    "/Game/MetaHumans/gakuchyou_v3/Body/SKM_gakuchyouv3_BodyMesh",
    "/Game/MetaHumans/gakuchyou_v3/Face/SKM_gakuchyouv3_FaceMesh",
    "/Game/MetaHumans/gakuchyou_v3/Clothing/gakuchyouv3_Outfits",
]
V1_SET = [
    "/Game/MetaHumans/gakuchyou/Body/SKM_gakuchyou_BodyMesh",
    "/Game/MetaHumans/gakuchyou/Face/SKM_gakuchyou_FaceMesh",
    "/Game/MetaHumans/gakuchyou/Clothing/gakuchyou_Outfits",
]

def export_merged(asset_paths, base_name, lods=(0,), bake_size=1024):
    """GLTFExporter のウェイト読み出しバグ(出力側ボーン番号 >255 なのにセクション内参照が 8bit だと
    ウェイト位置を誤る)を避けるため、16bit ボーンインデックスを使うメッシュを混ぜて
    SkeletalMergingLibrary で 1 メッシュにマージしてから書き出す(マージ後は LOD 全体が 16bit になる)。"""
    meshes = []
    for path in asset_paths:
        skm = unreal.EditorAssetLibrary.load_asset(path)
        if not skm:
            log(f"  asset not found: {path}")
            return
        describe_skm(skm)
        # 可変影響数レイアウト(>8 影響)だとエクスポータが負の添字で書き込んで落ちるため、
        # 全メッシュを固定 8 影響にしてからマージする(顔由来で LOD 全体が 16bit index になる)
        make_export_safe(skm, lods)
        meshes.append(skm)
    params = unreal.SkeletalMeshMergeParams()
    params.set_editor_property("meshes_to_merge", meshes)
    params.set_editor_property("needs_cpu_access", True)
    params.set_editor_property("strip_top_lods", 0)
    try:
        skels = []
        for m in meshes:
            sk = m.get_editor_property("skeleton")
            if sk and sk not in skels:
                skels.append(sk)
        if len(skels) > 1:
            sp = unreal.SkeletonMergeParams()
            sp.set_editor_property("skeletons_to_merge", skels)
            merged_skel = unreal.SkeletalMergingLibrary.merge_skeletons(sp)
            log(f"  merged skeleton: {merged_skel}")
            if merged_skel:
                params.set_editor_property("skeleton", merged_skel)
        else:
            params.set_editor_property("skeleton", skels[0])
    except Exception as e:  # noqa: BLE001
        log(f"  (skeleton merge skipped: {e})")
    merged = unreal.SkeletalMergingLibrary.merge_meshes(params)
    if not merged:
        log(f"  merge failed for {base_name}")
        return
    log(f"  merged mesh: {merged.get_name()}")
    describe_skm(merged)
    for lod in lods:
        do_export(merged, f"{base_name}_lod{lod}.glb", make_options(lod=lod, bake_size=bake_size))


COMBINED = "/Game/gakuchyou_CombinedSkelMesh"

log("=== start ===")
try:
    steps = os.environ.get("GAKUCHO_STEPS", "v3body,v3face").split(",")
    if "v3mergeC" in steps:  # 体+服+結合メッシュ(16bit 化のためのダミー; 後で除去)
        export_merged([V3_SET[0], V3_SET[2], COMBINED], "gakucho_v3_mergeC")
    if "v3mergeF" in steps:  # 体+服+顔(全部入り)
        export_merged([V3_SET[0], V3_SET[2], V3_SET[1]], "gakucho_v3_mergeF")
    if "v1mergeC" in steps:
        export_merged([V1_SET[0], V1_SET[2], COMBINED], "gakucho_v1_mergeC")
    if "v1mergeF" in steps:
        export_merged([V1_SET[0], V1_SET[2], V1_SET[1]], "gakucho_v1_mergeF")
    if "v3body" in steps:
        export_mesh_set([V3_SET[0], V3_SET[2]], "gakucho_v3_bodyoutfit", lods=(0,))
    if "v3face" in steps:
        export_mesh_set([V3_SET[1]], "gakucho_v3_face", lods=(0, 1))
    if "v1body" in steps:
        export_mesh_set([V1_SET[0], V1_SET[2]], "gakucho_v1_bodyoutfit", lods=(0,))
    if "v1face" in steps:
        export_mesh_set([V1_SET[1]], "gakucho_v1_face", lods=(0, 1))
    if "v3set" in steps:
        export_mesh_set(V3_SET, "gakucho_v3_set", lods=(0, 1))
    if "v1set" in steps:
        export_mesh_set(V1_SET, "gakucho_v1_set", lods=(0, 1))
    if "hair" in steps:
        export_static_mesh("/Game/MetaHumans/gakuchyou_v3/Grooms/Hair_S_SideSweptFringe_CardsMesh_Group0_LOD0", "gakucho_v3_haircards")
    if "actor" in steps:
        export_actor("/Game/BP_gakuchyou", "gakucho_actor")
    if "v3" in steps:
        export_actor("/Game/MetaHumans/gakuchyou_v3/BP_gakuchyou_v3", "gakucho_v3_actor")
    if "body" in steps:
        export_skm("/Game/MetaHumans/gakuchyou/Body/SKM_gakuchyou_BodyMesh", "gakucho_body", lods_to_try=(0, 2))
    if "combined" in steps:
        export_skm("/Game/gakuchyou_CombinedSkelMesh", "gakucho_combined", lods_to_try=(0, 1, 2))
except Exception:  # noqa: BLE001
    log("TOP-LEVEL EXCEPTION:\n" + traceback.format_exc())
log("=== done ===")
STATUS.close()
