# -*- coding: utf-8 -*-
"""작업자-장비 거리 오차 + 위험 단계(위험/경고/안전) 판정 평가.  (GPU 불필요, numpy만 사용)
거리 = 바닥면에 투영한 거리(m). 장비는 3D 박스 8개 꼭짓점을 바닥에 투영한 볼록 다각형(= 위에서 본 외곽)으로 보고,
'edge' 거리 = 작업자 중심 ~ 장비 외곽까지의 최단거리(안에 있으면 0), 'center' 거리 = 두 중심 사이 거리.
바닥 방향(중력)은 사진마다 정답 박스의 '아래축' 평균으로 잡고, 정답/예측에 똑같이 적용한다(고정 카메라 가정).
"""
import argparse, json, os, csv, math
import numpy as np

NAMES = ["worker", "dumptruck", "excavator", "concretemixer", "forklift", "crane", "loader", "bulldozer", "grader", "roadroller"]
LV = ["위험", "경고", "안전"]


def verts(c, d, R):
    w, h, l = d
    sx = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * l / 2; sy = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * h / 2; sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * w / 2
    return (np.array(R) @ np.stack([sx, sy, sz])).T + np.array(c)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def basis(g):
    g = g / np.linalg.norm(g); z = np.array([0, 0, 1.0]); e1 = np.cross(g, z)
    if np.linalg.norm(e1) < 1e-6: e1 = np.cross(g, np.array([1.0, 0, 0]))
    e1 /= np.linalg.norm(e1); e2 = np.cross(g, e1); return e1, e2


def hull(P):  # 2D 볼록 껍질 (monotone chain)
    P = sorted(set(map(tuple, np.round(P, 6)))); 
    if len(P) <= 2: return np.array(P)
    def cr(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    up = []
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def pt_poly(p, H):
    p = np.array(p)
    if len(H) < 3: return float(min(np.linalg.norm(p - h) for h in H))
    inside = all((H[(i + 1) % len(H)][0] - H[i][0]) * (p[1] - H[i][1]) - (H[(i + 1) % len(H)][1] - H[i][1]) * (p[0] - H[i][0]) >= 0 for i in range(len(H)))
    if inside: return 0.0
    best = 1e9
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]; ab = b - a; t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1); best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def cross_class_nms(P, thr):
    """같은 장비를 서로 다른 클래스로 여러 번 예측한 중복을 제거. 작업자끼리, 장비끼리 따로 (2D IoU>=thr 이면 점수 낮은 쪽 제거)."""
    out = []
    for is_w in (True, False):
        kept = []
        for p in sorted([p for p in P if (p["category_id"] == 0) == is_w], key=lambda p: -p["score"]):
            x, y, w, h = p["bbox"]; b = [x, y, x + w, y + h]
            if all(iou(b, [q["bbox"][0], q["bbox"][1], q["bbox"][0] + q["bbox"][2], q["bbox"][1] + q["bbox"][3]]) < thr for q in kept): kept.append(p)
        out += kept
    return out


def level(d, t1, t2): return 0 if d <= t1 else (1 if d <= t2 else 2)


def match(gts, preds, agnostic=False):
    """클래스가 같고 2D IoU>=0.5 인 것끼리 큰 IoU 순으로 1:1 매칭. returns {gt_idx: pred_idx}"""
    cand = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if (not agnostic) and p["category_id"] != g["category_id"]: continue
            x, y, w, h = p["bbox"]; v = iou(g["_box"], [x, y, x + w, y + h])
            if v >= 0.5: cand.append((v, gi, pi))
    cand.sort(reverse=True); mg, mp = {}, set()
    for v, gi, pi in cand:
        if gi in mg or pi in mp: continue
        mg[gi] = pi; mp.add(pi)
    return mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", default="output/dist_eval")
    ap.add_argument("--thr", default="0.05,0.1,0.2,0.3", help="예측 점수 기준(쉼표로 여러 개)")
    ap.add_argument("--t1", type=float, default=4.0, help="위험 거리(m)"); ap.add_argument("--t2", type=float, default=8.0, help="경고 거리(m)")
    ap.add_argument("--metric", choices=["edge", "center"], default="edge"); ap.add_argument("--worker-r", type=float, default=0.0, help="작업자 반지름(m)을 거리에서 뺌")
    ap.add_argument("--no-ignore", dest="use_ignore", action="store_false", help="무시 영역(평가에서 제외된 정답) 위의 예측도 오경보로 센다"); ap.add_argument("--equip-any", action="store_true", help="장비는 클래스가 달라도 위치(2D IoU)만 맞으면 같은 장비로 인정 (작업자는 그대로)"); ap.add_argument("--nms", type=float, default=0.0, help="0보다 크면 클래스를 무시한 중복 제거(2D IoU 기준, 예: 0.5) 적용"); ap.add_argument("--vis", type=float, default=0.1); ap.add_argument("--trunc", type=float, default=0.75)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gt = json.load(open(a.gt)); pred = json.load(open(a.pred)); thrs = [float(t) for t in a.thr.split(",")]
    gts_by, prs_by = {}, {}
    for an in gt["annotations"]: gts_by.setdefault(an["image_id"], []).append(an)
    for p in pred: prs_by.setdefault(p["image_id"], []).append(p)
    all_rows = {}
    for thr in thrs:
        rows = []; ign_total = 0; fa_kind = [0, 0, 0, 0]; false_alarm = 0; false_alarm_loose = 0; pred_alarm = 0; n_pairs_gt = 0
        for im in gt["images"]:
            G = [g for g in gts_by.get(im["id"], []) if g["valid3D"] and not g["behind_camera"] and g["visibility"] >= a.vis and g["truncation"] <= a.trunc]
            if not G: continue
            for g in G: g["_box"] = g["bbox2D_trunc"] if not all(v == -1 for v in g["bbox2D_trunc"]) else g["bbox2D_proj"]
            Pd = [p for p in prs_by.get(im["id"], []) if p["score"] >= thr]
            if a.nms > 0: Pd = cross_class_nms(Pd, a.nms)
            down = np.array([np.array(g["R_cam"])[:, 1] for g in G]); ref = down[0]; down = np.array([d if np.dot(d, ref) >= 0 else -d for d in down]); gvec = down.mean(0)
            e1, e2 = basis(gvec)
            def gp(P): P = np.atleast_2d(P); return np.stack([P @ e1, P @ e2], 1)
            Gids = {id(g) for g in G}; ign_boxes = []
            for g in gts_by.get(im["id"], []):
                if id(g) in Gids: continue
                for key in ("bbox2D_trunc", "bbox2D_proj", "bbox2D_tight"):
                    b = g.get(key)
                    if b and not all(v == -1 for v in b): ign_boxes.append(b); break
            gfoot = [hull(gp(verts(g["center_cam"], g["dimensions"], g["R_cam"]))) for g in G]; gcen = [gp(g["center_cam"])[0] for g in G]
            pfoot = [hull(gp(verts(p["center_cam"], p["dimensions"], p["pose"]))) for p in Pd]; pcen = [gp(p["center_cam"])[0] for p in Pd]
            if a.equip_any:
                gW = [i for i, g in enumerate(G) if g["category_id"] == 0]; gE = [i for i, g in enumerate(G) if g["category_id"] != 0]
                pW = [i for i, p in enumerate(Pd) if p["category_id"] == 0]; pE = [i for i, p in enumerate(Pd) if p["category_id"] != 0]
                mg = {gW[a_]: pW[b_] for a_, b_ in match([G[i] for i in gW], [Pd[i] for i in pW]).items()}
                mg.update({gE[a_]: pE[b_] for a_, b_ in match([G[i] for i in gE], [Pd[i] for i in pE], agnostic=True).items()})
            else: mg = match(G, Pd)
            gi_of_p = {pi: gi for gi, pi in mg.items()}
            ign_p = set()
            if a.use_ignore:
                for pi, p in enumerate(Pd):
                    if pi in gi_of_p: continue
                    x, y, w_, h_ = p["bbox"]
                    if any(iou([x, y, x + w_, y + h_], b) >= 0.5 for b in ign_boxes): ign_p.add(pi)
            ign_total += len(ign_p)
            def dist(wc, ec, ef):
                return max(0.0, (pt_poly(wc, ef) if a.metric == "edge" else float(np.linalg.norm(wc - ec))) - a.worker_r)
            W = [i for i, g in enumerate(G) if g["category_id"] == 0]; E = [i for i, g in enumerate(G) if g["category_id"] != 0]
            for wi in W:
                for ei in E:
                    dg = dist(gcen[wi], gcen[ei], gfoot[ei]); n_pairs_gt += 1
                    r = dict(image=os.path.basename(im["file_path"]), worker_gt=wi, equip_gt=ei, equip_cls=NAMES[G[ei]["category_id"]], d_gt=dg, lv_gt=level(dg, a.t1, a.t2), detected=0,
                             worker_matched=int(wi in mg), equip_matched=int(ei in mg), worker_z=G[wi]["center_cam"][2], equip_z=G[ei]["center_cam"][2], worker_h_px=G[wi]["_box"][3] - G[wi]["_box"][1])
                    if wi in mg and ei in mg:
                        dp = dist(pcen[mg[wi]], pcen[mg[ei]], pfoot[mg[ei]]); r.update(detected=1, d_pred=dp, lv_pred=level(dp, a.t1, a.t2), err=dp - dg)
                    rows.append(r)
            # 예측 쌍 중 경보(위험/경고)가 났는데 정답이 아닌 것 (오경보)
            PW = [i for i, p in enumerate(Pd) if p["category_id"] == 0]; PE = [i for i, p in enumerate(Pd) if p["category_id"] != 0]
            ml = match([G[i] for i in E], [Pd[i] for i in PE], agnostic=True); loose_p2g = {PE[lp]: E[lg] for lg, lp in ml.items()}  # 장비는 클래스 무시하고 매칭
            for pw_ in PW:
                for pe_ in PE:
                    dp = dist(pcen[pw_], pcen[pe_], pfoot[pe_]); lp = level(dp, a.t1, a.t2)
                    if lp == 2: continue
                    if pw_ in ign_p or pe_ in ign_p: continue  # 평가에서 제외된 객체(무시 영역)에 해당하는 예측
                    pred_alarm += 1
                    if pw_ in gi_of_p and pe_ in loose_p2g and level(dist(gcen[gi_of_p[pw_]], gcen[loose_p2g[pe_]], gfoot[loose_p2g[pe_]]), a.t1, a.t2) <= lp: pass
                    else: false_alarm_loose += 1
                    if pw_ in gi_of_p and pe_ in gi_of_p:
                        dg = dist(gcen[gi_of_p[pw_]], gcen[gi_of_p[pe_]], gfoot[gi_of_p[pe_]])
                        if level(dg, a.t1, a.t2) <= lp: continue  # 정답도 같거나 더 심한 단계 => 정당한 경보
                    false_alarm += 1; fa_kind[(1 if pw_ in gi_of_p else 0) + (2 if pe_ in gi_of_p else 0)] += 1
        all_rows[thr] = (rows, pred_alarm, false_alarm, n_pairs_gt)
        if thr == thrs[0]:
            print("\n[정답이 위험/경고인 쌍 목록 - 왜 놓쳤는지 확인용 (작업자/장비 매칭 여부, 작업자 2D 높이, 깊이)]")
            for r in sorted([r for r in rows if r["lv_gt"] <= 1], key=lambda r: (r["image"], r["d_gt"])):
                print(f"  {r['image'][-22:]:22s} {LV[r['lv_gt']]} 거리 {r['d_gt']:4.1f}m  장비 {r['equip_cls']:13s} z={r['equip_z']:3.0f}m 검출={'O' if r['equip_matched'] else 'X'} | 작업자 z={r['worker_z']:3.0f}m 높이 {r['worker_h_px']:3.0f}px 검출={'O' if r['worker_matched'] else 'X'}")
        # ---- 출력 ----
        D = [r for r in rows if r["detected"]]
        print(f"\n===== 점수>={thr}  (거리 기준: {a.metric}, 위험<={a.t1}m, 경고<={a.t2}m, 클래스무시 중복제거 IoU={a.nms if a.nms > 0 else '끔'}, 장비 클래스 무시 매칭={'켬' if a.equip_any else '끔'}) =====")
        print(f"정답 작업자-장비 쌍 {n_pairs_gt}개 중 두 객체가 모두 검출된 쌍 {len(D)}개 ({100*len(D)/max(n_pairs_gt,1):.0f}%)")
        for k, nm in enumerate(LV[:2]):
            gk = [r for r in rows if r["lv_gt"] == k]; hit = [r for r in gk if r["detected"] and r["lv_pred"] <= k]
            near = [r for r in gk if r["detected"] and r["lv_pred"] <= 1]
            print(f"  정답 {nm} 쌍 {len(gk):3d}개: 두 객체 검출 {sum(r['detected'] for r in gk):3d}, 그중 같은 단계 이상으로 판정 {len(hit):3d}  -> 재현율 {len(hit)/max(len(gk),1):.2f}")
        if D:
            err = np.array([r["err"] for r in D]); ae = np.abs(err)
            print(f"  거리 오차(검출된 쌍 {len(D)}개): 중앙값 |오차| {np.median(ae):.2f}m, 평균 부호오차 {err.mean():+.2f}m, 1m이내 {np.mean(ae<=1):.2f}, 2m이내 {np.mean(ae<=2):.2f}, 4m이내 {np.mean(ae<=4):.2f}")
            for lo, hi in [(0, a.t1), (a.t1, a.t2), (a.t2, 20), (20, 1e9)]:
                s = [r for r in D if lo < r["d_gt"] <= hi or (lo == 0 and r["d_gt"] == 0)]
                if s:
                    hi_s = ("%gm" % hi) if hi < 1e8 else "끝"
                    print(f"     실제 {lo:g}~{hi_s} 쌍 {len(s):4d}개: 중앙값 |오차| {np.median([abs(r['err']) for r in s]):.2f}m, 평균 부호오차 {np.mean([r['err'] for r in s]):+.2f}m")
            C = np.zeros((3, 3), int)
            for r in D: C[r["lv_gt"], r["lv_pred"]] += 1
            print("  단계 혼동표 (행=정답, 열=예측):  " + "  ".join(f"{x:>4s}" for x in LV))
            for i in range(3): print(f"        정답 {LV[i]}: " + "  ".join(f"{C[i,j]:4d}" for j in range(3)))
        print(f"  예측이 경보(위험/경고)를 낸 쌍 {pred_alarm}개 중 정답과 안 맞는 오경보 {false_alarm}개 (장비 클래스를 무시하고 보면 {false_alarm_loose}개)")
        if a.use_ignore: print(f"     (평가에서 제외된 객체 위에 놓인 예측 {ign_total}개는 오경보 계산에서 뺐음)")
        print(f"     오경보 {false_alarm}개의 원인: 작업자도 장비도 정답과 안 맞음(허위 검출 등) {fa_kind[0]}, 작업자만 맞음 {fa_kind[1]}, 장비만 맞음(작업자가 허위 검출) {fa_kind[2]}, 둘 다 맞았는데 거리가 틀려 위험 단계가 과대평가됨 {fa_kind[3]}")
    rows = all_rows[thrs[0]][0]
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, f"pairs_thr{thrs[0]}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print("\n쌍별 표:", os.path.join(a.out, f"pairs_thr{thrs[0]}.csv"))


if __name__ == "__main__":
    main()
