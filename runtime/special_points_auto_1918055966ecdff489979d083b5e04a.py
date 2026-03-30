import json
import sys
sys.path.insert(0, r'/media/kevin0412/Data/mandelbrot/C_mandelbrot')
import special_points_of_mandelbrot_set as sp
points = sp.solution(0,4)
with open(r'/media/kevin0412/Data/mandelbrot/fractal_studio/runtime/special_points_auto_1918055966ecdff489979d083b5e04a.json', 'w', encoding='utf-8') as f:
    json.dump([{'real': float(x.real), 'imag': float(x.imag)} for x in points], f, ensure_ascii=False)
