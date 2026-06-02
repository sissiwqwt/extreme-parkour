cd /home/rhos/jiamin/extreme-parkour/legged_gym/legged_gym/scripts

mkdir -p ../../../result/videos /tmp/base_play_variants

python - <<'PY'
from pathlib import Path

terrains = ["bean_gap", "asymmetric_gap", "parkour_v2", "narrow_gap", "climbing_wall"]
src = Path("base_play.py").read_text()

for active in terrains:
      s = src
      for t in terrains:
          s = s.replace(f'"{t}": 1.0,', f'"{t}": 0.0,')
          s = s.replace(f'"{t}": 0.0,', f'"{t}": {1.0 if t == active else 0.0},')
      Path(f"/tmp/base_play_variants/base_play_{active}.py").write_text(s)
PY

for terrain in bean_gap asymmetric_gap parkour_v2 narrow_gap climbing_wall; do
    python /tmp/base_play_variants/base_play_${terrain}.py \
      --task a1 \
      --proj_name ../../checkpoints \
      --exptid base \
      --checkpoint 20000 \
      --terrain_difficulty 1.0 \
      --video_out ../../../result/videos/${terrain}_base_20000_difficulty_1.0.mp4
done