"use client";

import { ChangeEvent, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { platform, type PayoutRequest } from "@/lib/api/platform";

function text(error: unknown): string { return error instanceof Error ? error.message : "Request failed"; }

export default function PayoutsPage() {
  const [rows, setRows] = useState<PayoutRequest[]>([]); const [amount, setAmount] = useState("10.00"); const [file, setFile] = useState<File | null>(null); const [handle, setHandle] = useState(""); const [displayName, setDisplayName] = useState(""); const [error, setError] = useState<string | null>(null);
  const refresh = () => void platform.payouts.list().then((value) => setRows(value.data)).catch((reason: unknown) => setError(text(reason)));
  useEffect(refresh, []);
  const request = async () => { if (!file) { setError("Choose payout QR code"); return; } try { await platform.payouts.create(amount, file); refresh(); } catch (reason) { setError(text(reason)); } };
  return <div className="space-y-5"><div><h1 className="text-2xl font-semibold">Creator payouts</h1><p className="text-muted-foreground">Upload Alipay receiving QR. Finance operator sends manual transfer.</p></div><section className="max-w-lg space-y-3 rounded-xl border border-white/10 p-4"><h2 className="font-medium">Become creator</h2><Input value={handle} placeholder="handle" onChange={(event) => setHandle(event.target.value)} /><Input value={displayName} placeholder="display name" onChange={(event) => setDisplayName(event.target.value)} /><Button variant="outline" disabled={!handle || !displayName} onClick={() => void platform.auth.creatorProfile(handle, displayName).catch((reason: unknown) => setError(text(reason)))}>Save creator profile</Button></section><section className="max-w-lg space-y-3 rounded-xl border border-white/10 p-4"><Input value={amount} onChange={(event) => setAmount(event.target.value)} placeholder="Amount CNY" /><input type="file" accept="image/png,image/jpeg" onChange={(event: ChangeEvent<HTMLInputElement>) => setFile(event.target.files?.[0] ?? null)} /><Button onClick={() => void request()}>Request payout</Button></section>{error && <p className="text-red-400">{error}</p>}<div className="space-y-2">{rows.map((row) => <article key={row.id} className="rounded border border-white/10 p-3"><b>{row.amount} {row.currency}</b> · {row.status}{row.status === "pending" && <Button size="sm" variant="outline" className="ml-3" onClick={() => void platform.payouts.cancel(row.id).then(refresh).catch((reason: unknown) => setError(text(reason)))}>Cancel</Button>}</article>)}</div></div>;
}
