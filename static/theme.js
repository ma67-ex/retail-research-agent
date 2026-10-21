(function () {
  var KEY = 'rra.theme';
  var THEMES = ['auto', 'light', 'dark', 'space'];
  var META = { light: '#e9edf1', dark: '#090c11', space: '#04060c' };
  var root = document.documentElement;
  var dark = matchMedia('(prefers-color-scheme: dark)');
  var still = matchMedia('(prefers-reduced-motion: reduce)');
  var fine = matchMedia('(hover: hover) and (pointer: fine)');
  var forced = new URLSearchParams(location.search).get('theme');

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function choice() {
    var c = THEMES.indexOf(forced) >= 0 ? forced : stored();
    return THEMES.indexOf(c) >= 0 ? c : 'auto';
  }
  function resolve(c) {
    return c === 'auto' ? (dark.matches ? 'dark' : 'light') : c;
  }
  function apply(c) {
    var t = resolve(c);
    root.dataset.theme = t;
    root.dataset.themeChoice = c;
    var meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.name = 'theme-color';
      document.head.appendChild(meta);
    }
    meta.content = META[t];
    document.dispatchEvent(new CustomEvent('rra:theme', { detail: { choice: c, theme: t } }));
  }

  apply(choice());
  dark.addEventListener('change', function () { if (choice() === 'auto') apply('auto'); });

  window.RRTheme = {
    themes: THEMES,
    choice: choice,
    set: function (c) {
      if (THEMES.indexOf(c) < 0) return;
      forced = null;
      try { localStorage.setItem(KEY, c); } catch (e) {}
      if (document.startViewTransition && !still.matches) document.startViewTransition(function () { apply(c); });
      else apply(c);
    },
  };

  // Seeded so the sky looks the same on every load.
  function rng(seed) {
    return function () {
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function tile(size, count, rMin, rMax, seed, halo) {
    var r = rng(seed), out = '';
    for (var i = 0; i < count; i++) {
      var x = (r() * size).toFixed(1), y = (r() * size).toFixed(1);
      var rad = rMin + r() * (rMax - rMin), o = 0.35 + r() * 0.65;
      var tint = r() > 0.8 ? '#cfe0ff' : '#ffffff';
      if (halo && r() > 0.45) out += "<circle cx='" + x + "' cy='" + y + "' r='" + (rad * 3.4).toFixed(2) + "' fill='" + tint + "' opacity='" + (o * 0.13).toFixed(3) + "'/>";
      out += "<circle cx='" + x + "' cy='" + y + "' r='" + rad.toFixed(2) + "' fill='" + tint + "' opacity='" + o.toFixed(2) + "'/>";
    }
    var svg = "<svg xmlns='http://www.w3.org/2000/svg' width='" + size + "' height='" + size + "'>" + out + '</svg>';
    return 'url("data:image/svg+xml,' + encodeURIComponent(svg) + '")';
  }

  function buildSky() {
    if (document.querySelector('.sky')) return;
    var sky = document.createElement('div');
    sky.className = 'sky';
    sky.setAttribute('aria-hidden', 'true');
    var specs = [[520, 120, 0.35, 0.75, 7, false], [760, 50, 0.55, 1.05, 13, false], [1040, 18, 0.8, 1.4, 29, true]];
    var layers = specs.map(function (s, i) {
      var l = document.createElement('div');
      l.className = 'stars l' + (i + 1);
      l.style.backgroundImage = tile(s[0], s[1], s[2], s[3], s[4], s[5]);
      sky.appendChild(l);
      return l;
    });
    ['m1', 'm2'].forEach(function (m) {
      var e = document.createElement('div');
      e.className = 'meteor ' + m;
      sky.appendChild(e);
    });
    document.body.insertBefore(sky, document.body.firstChild);

    if (!fine.matches || still.matches) return;
    var depth = [6, 14, 26], frame = 0, px = 0, py = 0;
    addEventListener('pointermove', function (e) {
      px = e.clientX / innerWidth - 0.5;
      py = e.clientY / innerHeight - 0.5;
      if (frame) return;
      frame = requestAnimationFrame(function () {
        frame = 0;
        if (root.dataset.theme === 'light') return;
        layers.forEach(function (l, i) {
          l.style.transform = 'translate3d(' + (-px * depth[i]).toFixed(1) + 'px,' + (-py * depth[i]).toFixed(1) + 'px,0)';
        });
      });
    }, { passive: true });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', buildSky);
  else buildSky();
})();
