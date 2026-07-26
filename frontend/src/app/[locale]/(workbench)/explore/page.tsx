"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { platform, submitAlipayForm, type Listing } from "@/lib/api/platform";

function text(error: unknown): string {
  return error instanceof Error ? error.message : "Request failed";
}

export default function ExplorePage() {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<Listing[]>([]);
  const [error, setError] = useState<string | null>(null);

  const search = () =>
    void platform.marketplace
      .explore(query)
      .then((value) => setItems(value.data))
      .catch((reason: unknown) => setError(text(reason)));

  useEffect(search, []);

  const checkout = async (listing: Listing) => {
    try {
      submitAlipayForm((await platform.commerce.checkout(listing)).alipayForm);
    } catch (reason) {
      setError(text(reason));
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Marketplace</h1>
        <p className="text-muted-foreground">Published art. Payment state is verified by Alipay webhook.</p>
      </div>
      <div className="flex gap-2">
        <Input value={query} placeholder="Search listings" onChange={(event) => setQuery(event.target.value)} />
        <Button onClick={search}>Search</Button>
      </div>
      {error && <p className="text-red-400">{error}</p>}
      {!error && items.length === 0 && (
        <p className="rounded-xl border border-dashed border-white/15 p-6 text-sm text-muted-foreground">
          No published listings found. Drafts are private: open My listings and publish a draft before it appears here.
        </p>
      )}
      <div className="grid grid-cols-2 gap-5">
        {items.map((listing) => (
          <article key={listing.id} className="min-w-0 overflow-hidden rounded-xl border border-white/10">
            <div className="aspect-[4/3] bg-white/5">
              {listing.preview?.thumbnailUrl ? (
                <img src={listing.preview.thumbnailUrl} alt={`Preview of ${listing.title}`} className="block h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center p-3 text-center text-sm text-muted-foreground">
                  Preview unavailable
                </div>
              )}
            </div>
            <div className="space-y-3 p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="truncate font-medium">{listing.title}</h2>
                  <p className="text-sm text-muted-foreground">by {listing.creator.displayName} · {listing.price} CNY</p>
                </div>
                <Button size="sm" variant="outline" onClick={() => void platform.marketplace.favorite(listing.assetId).catch((reason: unknown) => setError(text(reason)))}>
                  Favorite
                </Button>
              </div>
              {listing.description && <p className="line-clamp-2 text-sm text-muted-foreground">{listing.description}</p>}
              <Button className="w-full" onClick={() => void checkout(listing)}>Pay with Alipay</Button>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
