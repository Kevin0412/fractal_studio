"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { platform, submitAlipayForm, type Listing } from "@/lib/api/platform";

function text(error: unknown): string { return error instanceof Error ? error.message : "Request failed"; }

export default function ExplorePage() {
  const [query, setQuery] = useState(""); const [items, setItems] = useState<Listing[]>([]); const [error, setError] = useState<string | null>(null);
  const search = () => void platform.marketplace.explore(query).then((value) => setItems(value.data)).catch((reason: unknown) => setError(text(reason)));
  useEffect(search, []);
  const checkout = async (listing: Listing) => { try { submitAlipayForm((await platform.commerce.checkout(listing)).alipayForm); } catch (reason) { setError(text(reason)); } };
  return <div className="space-y-5"><div><h1 className="text-2xl font-semibold">Marketplace</h1><p className="text-muted-foreground">Published art. Payment state is verified by Alipay webhook.</p></div><div className="flex gap-2"><Input value={query} placeholder="Search listings" onChange={(event) => setQuery(event.target.value)} /><Button onClick={search}>Search</Button></div>{error && <p className="text-red-400">{error}</p>}{!error && items.length === 0 && <p className="rounded-xl border border-dashed border-white/15 p-6 text-sm text-muted-foreground">No published listings found. Drafts are private: open My listings and publish a draft before it appears here.</p>}<div className="grid gap-3 md:grid-cols-2">{items.map((listing) => <article key={listing.id} className="rounded-xl border border-white/10 p-4"><div className="flex items-start justify-between gap-3"><div><b>{listing.title}</b><p className="text-sm text-muted-foreground">by {listing.creator.displayName} · {listing.price} CNY</p></div><Button size="sm" variant="outline" onClick={() => void platform.marketplace.favorite(listing.assetId).catch((reason: unknown) => setError(text(reason)))}>Favorite</Button></div>{listing.preview?.thumbnailUrl && <img src={listing.preview.thumbnailUrl} alt="Listing preview" className="mt-3 max-h-48 rounded" />}<p className="mt-3 text-sm">{listing.description}</p><Button className="mt-3" onClick={() => void checkout(listing)}>Pay with Alipay</Button></article>)}</div></div>;
}
