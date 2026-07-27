"use client";

import { ChangeEvent, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/toaster";
import { useAuth } from "@/providers/auth-provider";
import { authKeys } from "@/lib/hooks/use-auth";
import { platform, PlatformApiError, type CreatorBalance, type PayoutRequest } from "@/lib/api/platform";

function text(error: unknown): string {
  return error instanceof Error ? error.message : "Request failed";
}

export default function PayoutsPage() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [rows, setRows] = useState<PayoutRequest[]>([]);
  const [balance, setBalance] = useState<CreatorBalance | null>(null);
  const [amount, setAmount] = useState("10.00");
  const [file, setFile] = useState<File | null>(null);
  const [handle, setHandle] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [requestingPayout, setRequestingPayout] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () =>
    void Promise.all([platform.payouts.list(), platform.payouts.balance()])
      .then(([requests, creatorBalance]) => { setRows(requests.data); setBalance(creatorBalance); })
      .catch((reason: unknown) => setError(text(reason)));

  const isCreator = user?.roles.includes("creator") ?? false;
  useEffect(() => { if (isCreator) refresh(); }, [isCreator]);
  const pendingRequest = rows.find((row) => row.status === "pending");
  const availableBalance = Number(balance?.availableAmount ?? 0);
  const requestedAmount = Number(amount);
  const insufficientBalance = !Number.isFinite(requestedAmount) || requestedAmount <= 0 || requestedAmount > availableBalance;

  const saveCreatorProfile = async () => {
    const normalizedHandle = handle.trim();
    const normalizedDisplayName = displayName.trim();
    if (!/^[a-z0-9_]{3,32}$/.test(normalizedHandle)) {
      setError("Handle: 3–32 lowercase letters, numbers, or underscores.");
      return;
    }
    if (!normalizedDisplayName) {
      setError("Enter a display name.");
      return;
    }
    setSavingProfile(true);
    setError(null);
    try {
      const user = await platform.auth.creatorProfile(normalizedHandle, normalizedDisplayName);
      queryClient.setQueryData(authKeys.me, user);
      toast({
        title: "Creator profile created",
        description: `You can now publish listings as @${user.creatorProfile?.handle ?? normalizedHandle}.`,
        variant: "success",
      });
      refresh();
    } catch (reason) {
      setError(text(reason));
    } finally {
      setSavingProfile(false);
    }
  };

  const request = async () => {
    if (pendingRequest) {
      setError("You already have a payout request pending. Cancel it before creating another one.");
      return;
    }
    if (!file) {
      setError("Choose payout QR code");
      return;
    }
    setRequestingPayout(true);
    setError(null);
    try {
      await platform.payouts.create(amount, file);
      refresh();
    } catch (reason) {
      if (reason instanceof PlatformApiError && reason.code === "payout_request_pending") {
        setError("You already have a payout request pending. Cancel it before creating another one.");
      } else if (reason instanceof PlatformApiError && reason.code === "insufficient_creator_balance") {
        setError("Payout amount exceeds your available creator balance.");
      } else {
        setError(text(reason));
      }
      refresh();
    } finally {
      setRequestingPayout(false);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Creator payouts</h1>
        <p className="text-muted-foreground">Upload Alipay receiving QR. Finance operator sends manual transfer.</p>
      </div>
      {!isCreator && <section className="max-w-lg space-y-3 rounded-xl border border-white/10 p-4">
        <h2 className="font-medium">Become creator</h2>
        <Input
          value={handle}
          placeholder="handle (lowercase, e.g. fractal_artist)"
          maxLength={32}
          onChange={(event) => setHandle(event.target.value.toLowerCase())}
        />
        <Input value={displayName} placeholder="display name" maxLength={120} onChange={(event) => setDisplayName(event.target.value)} />
        <Button variant="outline" loading={savingProfile} disabled={!handle || !displayName} onClick={() => void saveCreatorProfile()}>
          Save creator profile
        </Button>
      </section>}
      {isCreator && <section className="max-w-lg space-y-3 rounded-xl border border-white/10 p-4">
        <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3 text-sm"><span className="text-muted-foreground">Available to withdraw</span><p className="mt-1 text-lg font-medium">{balance ? `${balance.availableAmount} ${balance.currency}` : "Loading balance…"}</p>{balance && availableBalance <= 0 && <p className="mt-1 text-xs text-muted-foreground">Funds appear here after a buyer pays for one of your listings.</p>}</div>
        <Input value={amount} onChange={(event) => setAmount(event.target.value)} placeholder="Amount CNY" disabled={!balance || availableBalance <= 0} />
        <input type="file" accept="image/png,image/jpeg" onChange={(event: ChangeEvent<HTMLInputElement>) => setFile(event.target.files?.[0] ?? null)} />
        {pendingRequest && <p className="rounded-lg border border-amber-400/25 bg-amber-400/10 p-3 text-sm text-amber-200">A payout request is already pending. Cancel it below if you need to submit a new one.</p>}
        {balance && insufficientBalance && !pendingRequest && <p className="text-sm text-amber-200">Enter an amount no greater than your available balance.</p>}
        <Button loading={requestingPayout} disabled={!file || Boolean(pendingRequest) || !balance || insufficientBalance} onClick={() => void request()}>{pendingRequest ? "Payout pending" : "Request payout"}</Button>
      </section>}
      {error && <p className="text-red-400">{error}</p>}
      <div className="space-y-2">
        {rows.map((row) => (
          <article key={row.id} className="rounded border border-white/10 p-3">
            <b>{row.amount} {row.currency}</b> · {row.status}
            {row.status === "pending" && (
              <Button
                size="sm"
                variant="outline"
                className="ml-3"
                onClick={() => void platform.payouts.cancel(row.id).then(refresh).catch((reason: unknown) => setError(text(reason)))}
              >
                Cancel
              </Button>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
