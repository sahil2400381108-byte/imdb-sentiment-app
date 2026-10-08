/* One-time entrances and pointer-driven artwork; content is visible without JS. */
(() => {
  const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  const running = new Set();
  const seen = new WeakSet();
  const hero = document.querySelector(".cinema-hero");
  let pointerFrame = 0;
  let observer;

  function resetHero() {
    cancelAnimationFrame(pointerFrame);
    pointerFrame = 0;
    hero.style.removeProperty("--art-x");
    hero.style.removeProperty("--art-y");
  }

  hero.addEventListener("pointermove", (event) => {
    if (preference.matches || !finePointer.matches || event.pointerType === "touch") return;
    const bounds = hero.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width - .5) * 12;
    const y = ((event.clientY - bounds.top) / bounds.height - .5) * 10;
    cancelAnimationFrame(pointerFrame);
    pointerFrame = requestAnimationFrame(() => {
      hero.style.setProperty("--art-x", `${x.toFixed(2)}px`);
      hero.style.setProperty("--art-y", `${y.toFixed(2)}px`);
      pointerFrame = 0;
    });
  }, {passive: true});
  hero.addEventListener("pointerleave", resetHero);
  hero.addEventListener("pointercancel", resetHero);

  const targets = document.querySelectorAll(".workspace-heading, .analyzer-grid > .card, .sample-card, .metric-grid, .how-it-works, .evaluation-grid > .card, .method-grid > .card, .methodology");
  function reveal(entries) {
    for (const entry of entries) {
      if (!entry.isIntersecting || preference.matches) continue;
      const element = entry.target;
      seen.add(element);
      observer.unobserve(element);
      // Vertical translation leaves borders, text and chart values crisp.
      const siblings = [...element.parentElement.children].filter(item => item.matches(".sample-card, .card"));
      const delay = Math.max(0, siblings.indexOf(element)) * 65;
      const animation = element.animate([
        {opacity: .15, translate: "0 18px"},
        {opacity: 1, translate: "0 0"}
      ], {duration: 560, delay, fill: "backwards", easing: "cubic-bezier(.16,1,.3,1)"});
      running.add(animation);
      animation.onfinish = animation.oncancel = () => running.delete(animation);
    }
  }
  if ("IntersectionObserver" in window && "animate" in Element.prototype) {
    observer = new IntersectionObserver(reveal, {threshold: .08});
  }

  function syncPreference() {
    resetHero();
    if (!observer) return;
    observer.disconnect();
    if (preference.matches) {
      for (const animation of running) animation.cancel();
      running.clear();
    } else {
      for (const target of targets) if (!seen.has(target)) observer.observe(target);
    }
  }
  preference.addEventListener("change", syncPreference);
  finePointer.addEventListener("change", resetHero);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      resetHero();
      for (const animation of running) animation.finish();
    }
  });
  syncPreference();
})();
