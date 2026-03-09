import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

const MODEL_ROWS = [
  {
    sceneId: '0a7cc',
    folder: '3d_visualize/0a7cc_optimized',
    manifest: 'manifest.json'
  },
  {
    sceneId: '09d6e808b4',
    folder: '3d_visualize/09d6e808b4_optimized',
    manifest: 'manifest.json'
  }
];

const viewers = [];
const viewerByContainer = new WeakMap();
const clock = new THREE.Clock();
let pageVisible = !document.hidden;
const ACTIVE_RENDER_INTERVAL_MS = 40;
const objLoader = new OBJLoader();
const basePromiseCache = new Map();
const manifestPromiseCache = new Map();

const CAMERA_PRESET = {
  padding: 1.18,
  distanceScale: 0.78,
  heightScale: 0.95,
  diagScale: 1.0
};

function makeCard({ title, badge, highlight }) {
  const card = document.createElement('div');
  card.className = `viewer-card rounded-2xl overflow-hidden transition-all duration-300 ${
    highlight ? 'glass-card card-highlight' : 'glass-card'
  }`;

  const header = document.createElement('div');
  header.className = 'flex items-center justify-between px-4 py-3 border-b border-white/5 bg-white/[0.02]';

  const hTitle = document.createElement('div');
  hTitle.className = 'text-sm font-semibold text-white';
  hTitle.textContent = title;

  const hBadge = document.createElement('div');
  if (highlight) {
    hBadge.className = 'text-[10px] px-2.5 py-1 rounded-full font-medium bg-gradient-to-r from-indigo-500/20 to-violet-500/20 border border-indigo-500/30 text-indigo-300';
  } else {
    hBadge.className = 'text-[10px] px-2.5 py-1 rounded-full font-medium bg-white/5 border border-white/10 text-slate-400';
  }
  hBadge.textContent = badge;

  header.appendChild(hTitle);
  header.appendChild(hBadge);

  const viewport = document.createElement('div');
  viewport.className = 'relative h-[280px]';

  const overlay = document.createElement('div');
  overlay.className = 'viewer-overlay';
  overlay.innerHTML = '<div class="text-center"><div class="spinner"></div><div class="text-slate-400 text-xs">Loading mesh…</div></div>';
  viewport.appendChild(overlay);

  card.appendChild(header);
  card.appendChild(viewport);

  return { card, viewport, overlay };
}

function fitCameraToObject(camera, controls, object, opts = {}) {
  const {
    padding = 1.15,
    distanceScale = 0.85,
    heightScale = 0.55,
    diagScale = 1.0
  } = opts;

  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const maxSize = Math.max(size.x, size.y, size.z);
  const fov = (camera.fov * Math.PI) / 180;
  let cameraZ = Math.abs((maxSize / 2) / Math.tan(fov / 2));
  cameraZ *= padding;

  const dist = cameraZ * distanceScale;
  const dir = new THREE.Vector3(diagScale, heightScale, diagScale).normalize();
  camera.position.copy(center).addScaledVector(dir, dist);
  camera.near = cameraZ / 100;
  camera.far = cameraZ * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

async function loadObj(url) {
  return await objLoader.loadAsync(url);
}

function cloneSceneObject(root) {
  return root.clone(true);
}

function getBaseUrl(folder, baseFile) {
  return `./${folder}/${baseFile}`;
}

function getDeltaUrl(folder, deltaFile) {
  return `./${folder}/${deltaFile}`;
}

async function loadManifest(folder, manifestFile) {
  const url = `./${folder}/${manifestFile}`;
  if (!manifestPromiseCache.has(url)) {
    const promise = fetch(url).then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} while loading manifest: ${url}`);
      }
      return response.json();
    });
    manifestPromiseCache.set(url, promise);
  }
  return await manifestPromiseCache.get(url);
}

async function loadBaseObject(baseUrl) {
  if (!basePromiseCache.has(baseUrl)) {
    const promise = loadObj(baseUrl).then((root) => {
      applyMaterial(root);
      return root;
    });
    basePromiseCache.set(baseUrl, promise);
  }
  return cloneSceneObject(await basePromiseCache.get(baseUrl));
}

async function loadDeltaObject(deltaUrl) {
  const root = await loadObj(deltaUrl);
  applyMaterial(root);
  return root;
}

function getVariantBadge(variant) {
  const key = String(variant && variant.key ? variant.key : '').toLowerCase();
  if (key === 'ours') return 'Holi-Spatial';
  if (key === 'gt') return 'Ground Truth';
  return 'Baseline';
}

function isOursVariant(variant) {
  return String(variant && variant.key ? variant.key : '').toLowerCase() === 'ours';
}

function normalizeVariant(variant) {
  return {
    title: variant && variant.title ? variant.title : (variant && variant.key ? variant.key : 'Unknown'),
    badge: getVariantBadge(variant),
    highlight: isOursVariant(variant),
    delta: variant && variant.delta ? variant.delta : ''
  };
}

function splitManifestVariants(manifest) {
  const variants = Array.isArray(manifest && manifest.variants)
    ? manifest.variants.map(normalizeVariant).filter((item) => item.delta)
    : [];
  const ours = variants.find((item) => item.highlight) || null;
  const baselines = variants.filter((item) => !item.highlight);
  return { baselines, ours };
}

function applyMaterial(root) {
  root.traverse((c) => {
    if (c && c.isMesh) {
      const hasVertexColors = !!(c.geometry && c.geometry.attributes && c.geometry.attributes.color);
      c.material = new THREE.MeshBasicMaterial({
        color: hasVertexColors ? 0xffffff : 0xe8ecf8,
        vertexColors: hasVertexColors,
        side: THREE.DoubleSide
      });
      c.castShadow = false;
      c.receiveShadow = false;
    }
  });
}

function createViewer({ container, overlay, baseUrl, deltaUrl }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
  camera.position.set(2, 1.2, 2);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = false;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.screenSpacePanning = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;
  controls.minDistance = 0.1;
  controls.maxDistance = 2000;

  let lastUserActionAt = 0;
  const pauseAutoRotateMs = 1200;
  let visible = false;
  let forcedRender = true;
  let lastRenderAt = 0;

  controls.addEventListener('start', () => {
    lastUserActionAt = performance.now();
    controls.autoRotate = false;
  });
  controls.addEventListener('end', () => {
    lastUserActionAt = performance.now();
  });

  function resize() {
    const { width, height } = container.getBoundingClientRect();
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    forcedRender = true;
  }

  async function load() {
    try {
      const group = new THREE.Group();
      const [baseRoot, deltaRoot] = await Promise.all([
        loadBaseObject(baseUrl),
        loadDeltaObject(deltaUrl)
      ]);
      group.add(baseRoot);
      group.add(deltaRoot);
      scene.add(group);
      fitCameraToObject(camera, controls, group, CAMERA_PRESET);
      overlay.classList.add('viewer-overlay--hidden');
      forcedRender = true;
    } catch (e) {
      overlay.innerHTML = `<div class="text-center px-4 leading-relaxed">
        <div class="font-semibold text-white mb-2 text-sm">Load Failed</div>
        <div class="text-slate-400 text-xs">Use a local server to view OBJ meshes.</div>
        <div class="text-slate-500 text-[10px] mt-2 break-all font-mono">${deltaUrl}</div>
      </div>`;
      console.error('OBJ load failed:', { baseUrl, deltaUrl }, e);
    }
  }

  function update(now) {
    if (!visible || !pageVisible) return;
    if (!forcedRender && now - lastRenderAt < ACTIVE_RENDER_INTERVAL_MS) return;
    if (!controls.autoRotate && performance.now() - lastUserActionAt > pauseAutoRotateMs) {
      controls.autoRotate = true;
    }
    controls.update();
    renderer.render(scene, camera);
    forcedRender = false;
    lastRenderAt = now;
  }

  resize();
  return {
    container,
    resize,
    update,
    load,
    setVisible(nextVisible) {
      visible = !!nextVisible;
      if (visible) forcedRender = true;
    },
    markDirty() {
      forcedRender = true;
    }
  };
}

function addViewer(container, item, folder, baseFile) {
  const { card, viewport, overlay } = makeCard({
    title: item.title,
    badge: item.badge,
    highlight: item.highlight
  });
  container.appendChild(card);
  const baseUrl = getBaseUrl(folder, baseFile);
  const deltaUrl = getDeltaUrl(folder, item.delta);
  const viewer = createViewer({ container: viewport, overlay, baseUrl, deltaUrl });
  viewers.push(viewer);
  viewerByContainer.set(viewer.container, viewer);
}

async function buildUI() {
  for (const row of MODEL_ROWS) {
    const manifest = await loadManifest(row.folder, row.manifest);
    const { baselines, ours } = splitManifestVariants(manifest);
    const baseFile = manifest && manifest.base ? manifest.base : 'base.obj';
    const baselinesEl = document.getElementById(`baselines-${row.sceneId}`);
    const oursEl = document.getElementById(`ours-${row.sceneId}`);

    if (baselinesEl) {
      for (const item of baselines) {
        addViewer(baselinesEl, item, row.folder, baseFile);
      }
    }

    if (oursEl && ours) {
      addViewer(oursEl, ours, row.folder, baseFile);
    }
  }
}

async function loadAllWithConcurrency(limit = 2) {
  let i = 0;
  const running = new Set();

  async function runOne(v) {
    const p = v.load().finally(() => running.delete(p));
    running.add(p);
    await p;
  }

  while (i < viewers.length) {
    while (running.size < limit && i < viewers.length) {
      runOne(viewers[i]);
      i++;
    }
    await Promise.race(running);
  }
  await Promise.allSettled([...running]);
}

function animate(now = performance.now()) {
  for (const v of viewers) v.update(now);
  requestAnimationFrame(animate);
}

function initViewerVisibility() {
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const viewer = viewerByContainer.get(entry.target);
          if (viewer) viewer.setVisible(entry.isIntersecting && entry.intersectionRatio > 0.1);
        }
      },
      { threshold: [0, 0.1, 0.25], rootMargin: '80px 0px 80px 0px' }
    );
    for (const v of viewers) observer.observe(v.container);
  } else {
    for (const v of viewers) v.setVisible(true);
  }

  document.addEventListener('visibilitychange', () => {
    pageVisible = !document.hidden;
    if (pageVisible) {
      for (const v of viewers) v.markDirty();
    }
  });
}

async function init() {
  await buildUI();
  initViewerVisibility();
  loadAllWithConcurrency(2);
  animate();
}

init().catch((error) => {
  console.error('Viewer initialization failed:', error);
});

let resizeRaf = 0;
window.addEventListener('resize', () => {
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    for (const v of viewers) v.resize();
  });
});
