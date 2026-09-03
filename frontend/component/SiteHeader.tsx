export default function SiteHeader() {
  return (
    <header className="border-b border-white/10 bg-[#091827]">
      <div className="mx-auto max-w-7xl px-6 py-4">
        <div className="flex items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div className="text-lg font-semibold tracking-tight text-white">
            sudheer<span className="text-cyan-400">.Dev</span>
          </div>

          <div className="text-right text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            Quantum Intelligence Lab
          </div>
        </div>

        <div className="py-5">
          <p className="text-xs font-medium uppercase tracking-[0.3em] text-cyan-400">
            AI / Quantum Research Platform
          </p>

          <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">
            AI-Powered Quantum Error Correction
          </h1>

          <p className="mt-2 max-w-3xl text-sm text-slate-400 md:text-base">
            Classical simulation platform for quantum error correction,
            noisy quantum experiments, and AI decoding.
          </p>
        </div>
      </div>
    </header>
  );
}