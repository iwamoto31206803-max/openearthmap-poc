python -c "from pathlib import Path; import hashlib; [print(p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in Path('training_outputs').rglob('best.pth')]"
