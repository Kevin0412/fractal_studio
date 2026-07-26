"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { platform, type FractalSpec, type Recipe, type RenderJob } from "@/lib/api/platform";

const defaults: FractalSpec = {
  version: 1,
  centerRe: -0.75,
  centerIm: 0,
  scale: 3,
  iterations: 256,
  variant: "mandelbrot",
  colorMap: "classic_cos",
  julia: false,
  bailout: 4,
  engine: "auto",
  scalarType: "auto",
};

function message(error: unknown): string {
  return error instanceof Error ? error.message : "Request failed";
}

export default function StudioPage() {
  const [spec, setSpec] = useState<FractalSpec>(defaults);
  const [preview, setPreview] = useState<string | null>(null);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [job, setJob] = useState<RenderJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canonical = useMemo(() => ({ ...spec, centerRe: Number(spec.centerRe), centerIm: Number(spec.centerIm), scale: Number(spec.scale), iterations: Number(spec.iterations) }), [spec]);

  useEffect(() => {
    void platform.studio.recipes().then((value) => setRecipes(value.data)).catch((reason: unknown) => setError(message(reason)));
  }, []);

  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      void platform.studio.job(job.id).then(setJob).catch((reason: unknown) => setError(message(reason)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  const update = (name: keyof FractalSpec, value: string | number | boolean) => setSpec((current) => ({ ...current, [name]: value }));

  const renderPreview = async () => {
    setBusy(true); setError(null);
    try {
      if (preview) URL.revokeObjectURL(preview);
      setPreview(URL.createObjectURL(await platform.studio.preview(canonical)));
    } catch (reason) { setError(message(reason)); }
    finally { setBusy(false); }
  };

  const saveAndRender = async () => {
    setBusy(true); setError(null);
    try {
      const recipe = await platform.studio.createRecipe(canonical);
      setRecipes((current) => [recipe, ...current.filter((item) => item.id !== recipe.id)]);
      setJob(await platform.studio.createRender(recipe.id, { kind: "image", format: "png", width: 1024, height: 1024 }));
    } catch (reason) { setError(message(reason)); }
    finally { setBusy(false); }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[20rem_1fr]">
      <section className="space-y-3 rounded-xl border border-white/10 bg-white/[0.02] p-4">
        <h1 className="text-xl font-semibold">Platform Studio</h1>
        <p className="text-sm text-muted-foreground">Preview stays bounded. Durable output is created through Platform worker.</p>
        <label className="block text-sm">Center real<Input value={spec.centerRe ?? 0} type="number" step="0.0001" onChange={(event) => update("centerRe", Number(event.target.value))} /></label>
        <label className="block text-sm">Center imaginary<Input value={spec.centerIm ?? 0} type="number" step="0.0001" onChange={(event) => update("centerIm", Number(event.target.value))} /></label>
        <label className="block text-sm">Scale<Input value={spec.scale ?? 3} type="number" min="0.000001" step="0.1" onChange={(event) => update("scale", Number(event.target.value))} /></label>
        <label className="block text-sm">Iterations<Input value={spec.iterations ?? 256} type="number" min="1" max="1000000" onChange={(event) => update("iterations", Number(event.target.value))} /></label>
        <label className="block text-sm">Variant<Input value={spec.variant ?? "mandelbrot"} onChange={(event) => update("variant", event.target.value)} /></label>
        <label className="block text-sm">Color map<Input value={spec.colorMap ?? "classic_cos"} onChange={(event) => update("colorMap", event.target.value)} /></label>
        <div className="flex gap-2"><Button onClick={() => void renderPreview()} disabled={busy}>Preview</Button><Button variant="outline" onClick={() => void saveAndRender()} disabled={busy}>Save + render PNG</Button></div>
        {error && <p className="text-sm text-red-400">{error}</p>}
      </section>
      <section className="space-y-5">
        <div className="min-h-80 rounded-xl border border-white/10 bg-black/30 p-3">
          {preview ? <img src={preview} alt="Fractal preview" className="mx-auto max-h-[40rem] max-w-full" /> : <p className="p-8 text-center text-muted-foreground">Create preview</p>}
        </div>
        {job && <div className="rounded-xl border border-white/10 p-4"><b>Render job</b><p>{job.status} · {job.progressPercent}%</p>{job.assetId && <p className="text-emerald-400">Asset ready: {job.assetId}</p>}{!["completed", "failed", "cancelled"].includes(job.status) && <Button size="sm" variant="outline" onClick={() => void platform.studio.cancel(job.id).then(setJob).catch((reason: unknown) => setError(message(reason)))}>Cancel</Button>}</div>}
        <div><h2 className="mb-2 text-lg font-medium">Saved recipes</h2><ul className="space-y-2">{recipes.map((recipe) => <li key={recipe.id} className="rounded border border-white/10 p-3 text-sm">{recipe.id} · {recipe.canonicalSpec.variant} · {recipe.canonicalSpec.colorMap ?? "default"}</li>)}</ul></div>
      </section>
    </div>
  );
}
