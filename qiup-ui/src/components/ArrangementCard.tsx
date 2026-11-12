"use client";

import { Check } from "lucide-react";
import type { ProfileInfo } from "@/lib/types";
import { Badge, cx } from "./ui";

function Schematic({ profile }: { profile: ProfileInfo }) {
  const two = profile.architecture === "two_crystal";
  const far = profile.imaging_mode === "far_field";
  return (
    <svg viewBox="0 0 220 64" className="h-14 w-full">
      <defs>
        <linearGradient id={`pump-${profile.id}`} x1="0" x2="1">
          <stop offset="0" stopColor="#3ee6a8" stopOpacity="0.2" />
          <stop offset="1" stopColor="#3ee6a8" />
        </linearGradient>
      </defs>
      <line x1="6" y1="32" x2="44" y2="32" stroke={`url(#pump-${profile.id})`} strokeWidth="2.5" />
      <circle cx="6" cy="32" r="3" fill="#3ee6a8" />
      {two ? (
        <>
          <rect x="44" y="22" width="22" height="20" rx="3" fill="#1c2538" stroke="#7c5cff" strokeWidth="1.2" />
          <text x="55" y="35" textAnchor="middle" fontSize="7" fill="#c4cce0">NL1</text>
          <line x1="66" y1="28" x2="146" y2="28" stroke="#7c5cff" strokeWidth="1.4" strokeDasharray="3 2" />
          <line x1="66" y1="36" x2="146" y2="36" stroke="#3ee6a8" strokeWidth="1.2" opacity="0.6" />
          <ellipse cx={far ? 86 : 90} cy="32" rx="3" ry="11" fill="none" stroke="#38bdf8" strokeWidth="1" />
          {far && <ellipse cx="126" cy="32" rx="3" ry="11" fill="none" stroke="#38bdf8" strokeWidth="1" />}
          <rect x="103" y="16" width="5" height="32" rx="1" fill="#ffffff" opacity="0.9" />
          <text x="105.5" y="12" textAnchor="middle" fontSize="7" fill="#97a3bd">T</text>
          <rect x="146" y="22" width="22" height="20" rx="3" fill="#1c2538" stroke="#7c5cff" strokeWidth="1.2" />
          <text x="157" y="35" textAnchor="middle" fontSize="7" fill="#c4cce0">NL2</text>
          <line x1="168" y1="32" x2="200" y2="32" stroke="#ff4d6d" strokeWidth="1.8" />
        </>
      ) : (
        <>
          <rect x="60" y="22" width="26" height="20" rx="3" fill="#1c2538" stroke="#7c5cff" strokeWidth="1.2" />
          <text x="73" y="35" textAnchor="middle" fontSize="7" fill="#c4cce0">NL</text>
          <line x1="44" y1="32" x2="60" y2="32" stroke="#3ee6a8" strokeWidth="2.5" />
          <line x1="86" y1="28" x2="160" y2="28" stroke="#7c5cff" strokeWidth="1.4" strokeDasharray="3 2" />
          <line x1="86" y1="36" x2="160" y2="36" stroke="#7c5cff" strokeWidth="1.4" strokeDasharray="3 2" opacity="0.55" />
          <ellipse cx="118" cy="32" rx="3" ry="11" fill="none" stroke="#38bdf8" strokeWidth="1" />
          {far && <text x="118" y="56" textAnchor="middle" fontSize="6" fill="#38bdf8">f</text>}
          <rect x="160" y="16" width="5" height="32" rx="1" fill="#ffffff" opacity="0.9" />
          <text x="162.5" y="12" textAnchor="middle" fontSize="7" fill="#97a3bd">R</text>
          <path d="M60 36 C 50 44, 40 50, 30 52" stroke="#ff4d6d" strokeWidth="1.8" fill="none" />
        </>
      )}
      <rect x={two ? 200 : 14} y={two ? 24 : 46} width="14" height={two ? 16 : 12} rx="2" fill="#0a0e18" stroke="#ff4d6d" strokeWidth="1.2" />
    </svg>
  );
}

export function ArrangementCard({
  profile,
  selected,
  onToggle,
}: {
  profile: ProfileInfo;
  selected: boolean;
  onToggle: () => void;
}) {
  const two = profile.architecture === "two_crystal";
  const far = profile.imaging_mode === "far_field";
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cx(
        "group relative overflow-hidden rounded-xl p-3 text-left transition-all",
        selected
          ? "bg-gradient-to-br from-sky-500/15 via-ink-800 to-violet-500/10 ring-1 ring-sky-400/50 shadow-[0_0_24px_-8px_rgba(56,189,248,0.6)]"
          : "bg-ink-900/70 ring-1 ring-white/5 hover:ring-white/15",
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span
            className={cx(
              "grid h-7 w-7 place-items-center rounded-lg font-mono text-sm font-bold",
              selected ? "bg-sky-400 text-ink-950" : "bg-ink-700 text-ink-200",
            )}
          >
            {profile.id}
          </span>
          <div>
            <div className="text-[12px] font-semibold text-white">{two ? "Two-crystal" : "Single-crystal"}</div>
            <div className="text-[10px] text-ink-400">{two ? "SU(1,1) NL1 → NL2" : "Double pass through NL"}</div>
          </div>
        </div>
        <span
          className={cx(
            "grid h-4 w-4 place-items-center rounded border transition-colors",
            selected ? "border-sky-400 bg-sky-400 text-ink-950" : "border-ink-600",
          )}
        >
          {selected && <Check className="h-3 w-3" strokeWidth={3} />}
        </span>
      </div>
      <div className="my-1 opacity-90 transition-opacity group-hover:opacity-100">
        <Schematic profile={profile} />
      </div>
      <div className="flex flex-wrap gap-1">
        <Badge tone={far ? "violet" : "sky"}>{far ? "Far field" : "Near field"}</Badge>
        <Badge tone={two ? "green" : "amber"}>{two ? "Transmissive T" : "Reflective R"}</Badge>
      </div>
    </button>
  );
}
