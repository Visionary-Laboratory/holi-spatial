const IMAGE_PREFIX = 'qa_visualize/';

// ── Video Gallery Data Source ──
// Source of truth is static HTML cards in `#videoGrid`.

// ── Lightbox ──

const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const lightboxClose = document.getElementById('lightbox-close');
let bodyScrollLockCount = 0;
let isLightboxOpen = false;
let isVideoModalOpen = false;
let isMenuOpen = false;

function logWarn(scope, err) {
  console.warn('[Holi-Spatial]', scope, err);
}

function getQaDataUrl(basePath) {
  const version = window.QA_DATA_VERSION;
  if (typeof version === 'string' && version.trim()) {
    return basePath + '?v=' + encodeURIComponent(version.trim());
  }
  return basePath;
}

function lockBodyScroll() {
  bodyScrollLockCount += 1;
  if (bodyScrollLockCount === 1) {
    document.body.style.overflow = 'hidden';
  }
}

function unlockBodyScroll() {
  bodyScrollLockCount = Math.max(0, bodyScrollLockCount - 1);
  if (bodyScrollLockCount === 0) {
    document.body.style.overflow = '';
  }
}

function openLightbox(src) {
  lightboxImage.src = src;
  lightbox.classList.add('open');
  lightbox.setAttribute('aria-hidden', 'false');
  if (!isLightboxOpen) {
    isLightboxOpen = true;
    lockBodyScroll();
  }
}

function closeLightbox() {
  lightbox.classList.remove('open');
  lightbox.setAttribute('aria-hidden', 'true');
  lightboxImage.src = '';
  if (isLightboxOpen) {
    isLightboxOpen = false;
    unlockBodyScroll();
  }
}

if (lightboxClose) lightboxClose.addEventListener('click', closeLightbox);
if (lightbox) lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox(); });

// ── Video Modal ──

const videoModalBackdrop = document.getElementById('videoModalBackdrop');
const modalVideo = document.getElementById('modalVideo');
const modalTitle = document.getElementById('modalTitle');
const modalCloseBtn = document.getElementById('modalClose');

function openVideoModal(item) {
  if (!videoModalBackdrop || !modalVideo) return;
  modalVideo.src = item.src;
  modalVideo.load();
  const p = modalVideo.play();
  if (p && typeof p.then === 'function') {
    p.catch((err) => logWarn('modal video play failed', err));
  }
  if (modalTitle) modalTitle.textContent = item.label || item.src;
  videoModalBackdrop.classList.add('is-open');
  if (!isVideoModalOpen) {
    isVideoModalOpen = true;
    lockBodyScroll();
  }
}

function closeVideoModal() {
  if (!videoModalBackdrop || !modalVideo) return;
  videoModalBackdrop.classList.remove('is-open');
  modalVideo.pause();
  modalVideo.removeAttribute('src');
  if (isVideoModalOpen) {
    isVideoModalOpen = false;
    unlockBodyScroll();
  }
}

if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeVideoModal);
if (videoModalBackdrop) {
  videoModalBackdrop.addEventListener('click', (e) => {
    if (e.target === videoModalBackdrop) closeVideoModal();
  });
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && videoModalBackdrop && videoModalBackdrop.classList.contains('is-open')) closeVideoModal();
});

// ── Video Gallery Rendering ──

(function initVideoGallery() {
  const galleryVideoCountEl = document.getElementById('galleryVideoCount');
  const galleryGridEl = document.getElementById('videoGrid');
  if (!galleryGridEl) return;

  const cards = Array.from(galleryGridEl.querySelectorAll('.video-card'));
  if (galleryVideoCountEl && galleryVideoCountEl.hasAttribute('data-auto-count')) {
    galleryVideoCountEl.textContent = String(cards.length);
  }

  cards.forEach((card) => {
    const src = card.getAttribute('data-video-src');
    const label = card.getAttribute('data-video-label') || src || 'video';
    if (!src) return;
    card.addEventListener('click', () => openVideoModal({ src, label }));
  });
})();

// ── Details Slider ──

(function initDetailsSlider() {
  const DETAILS_SCENES = [
    { id: '0000', label: 'Scene 0000', grounding: './after/details/0000/0000-1.mp4', depth: './after/details/0000-depth/0000-depth-1.mp4' },
    { id: '0071', label: 'Scene 0071', grounding: './after/details/0071/0071-1.mp4', depth: './after/details/0071-depth/0071-depth-1.mp4' },
    { id: '0628', label: 'Scene 0628', grounding: './after/details/0628/0628-1.mp4', depth: './after/details/0628-depth/0628-depth-1.mp4' },
    { id: '0659', label: 'Scene 0659', grounding: './after/details/0659/0659-1.mp4', depth: './after/details/0659-depth/0659-depth-1.mp4' },
  ];

  const sliderEl = document.getElementById('detailsSlider');
  const prevBtn = document.getElementById('detailsPrev');
  const nextBtn = document.getElementById('detailsNext');
  const dotsEl = document.getElementById('detailsDots');
  if (!sliderEl || !dotsEl) return;

  let currentIndex = 0;
  const slideEls = [];
  const dotEls = [];

  function createSlide(scene) {
    const slide = document.createElement('div');
    slide.className = 'details-slide';
    slide.setAttribute('data-scene-id', scene.id);
    slide.innerHTML =
      '<div class="details-slide-inner">' +
      '<div class="details-slide-title">' + scene.label + '</div>' +
      '<div class="details-videos">' +
      '<div class="details-video-cell">' +
      '<video src="' + scene.grounding + '" playsinline muted loop preload="metadata" data-role="grounding" data-autoplay-managed="true"></video>' +
      '<div class="video-overlay"><div class="video-play-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg></div></div>' +
      '<span class="details-video-label">3D Grounding</span>' +
      '</div>' +
      '<div class="details-video-cell">' +
      '<video src="' + scene.depth + '" playsinline muted loop preload="metadata" data-role="depth" data-autoplay-managed="true"></video>' +
      '<div class="video-overlay"><div class="video-play-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg></div></div>' +
      '<span class="details-video-label">Depth Estimation</span>' +
      '</div></div></div>';
    return slide;
  }

  DETAILS_SCENES.forEach((scene) => {
    const slide = createSlide(scene);
    sliderEl.appendChild(slide);
    slideEls.push(slide);
  });

  DETAILS_SCENES.forEach((_, i) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'details-dot' + (i === 0 ? ' is-active' : '');
    dot.setAttribute('aria-label', 'Scene ' + (i + 1));
    dot.setAttribute('role', 'tab');
    dot.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
    dot.addEventListener('click', () => goToSlide(i));
    dotsEl.appendChild(dot);
    dotEls.push(dot);
  });

  function goToSlide(index) {
    if (index < 0 || index >= slideEls.length) return;
    currentIndex = index;
    sliderEl.style.transform = 'translateX(' + (-index * 100) + '%)';
    dotEls.forEach((d, i) => {
      d.classList.toggle('is-active', i === index);
      d.setAttribute('aria-selected', i === index ? 'true' : 'false');
    });
    if (prevBtn) prevBtn.disabled = index === 0;
    if (nextBtn) nextBtn.disabled = index === slideEls.length - 1;
    pauseAllSlides();
    playCurrentSlide();
  }

  function pauseAllSlides() {
    slideEls.forEach((slide) => {
      slide.querySelectorAll('video').forEach((v) => v.pause());
    });
  }

  function safePlay(videoEl) {
    if (!videoEl) return;
    videoEl.muted = true;
    videoEl.defaultMuted = true;
    videoEl.autoplay = true;
    videoEl.loop = true;
    videoEl.playsInline = true;
    videoEl.setAttribute('playsinline', '');
    const p = videoEl.play();
    if (p && typeof p.then === 'function') {
      p.catch((err) => logWarn('details video play failed', err));
    }
  }

  function playCurrentSlide() {
    const activeSlide = slideEls[currentIndex];
    if (!activeSlide) return;
    activeSlide.querySelectorAll('video').forEach((v) => {
      safePlay(v);
    });
  }

  function syncTwoVideos(masterVideo, slaveVideo) {
    if (!masterVideo || !slaveVideo) return;

    function alignSlaveToMaster(force = false) {
      if (force || Math.abs(slaveVideo.currentTime - masterVideo.currentTime) > 0.08) {
        slaveVideo.currentTime = masterVideo.currentTime;
      }
    }

    function playSlaveFollowingMaster() {
      alignSlaveToMaster(true);
      safePlay(slaveVideo);
    }

    masterVideo.addEventListener('loadedmetadata', () => alignSlaveToMaster(true));
    masterVideo.addEventListener('play', playSlaveFollowingMaster);
    masterVideo.addEventListener('seeked', () => alignSlaveToMaster(true));
    masterVideo.addEventListener('timeupdate', () => alignSlaveToMaster(false));
    masterVideo.addEventListener('pause', () => slaveVideo.pause());

    // If slave stalls while master is playing, recover it.
    slaveVideo.addEventListener('pause', () => {
      if (!masterVideo.paused) safePlay(slaveVideo);
    });
  }

  slideEls.forEach((slide) => {
    const groundingVideo = slide.querySelector('video[data-role="grounding"]');
    const depthVideo = slide.querySelector('video[data-role="depth"]');
    [groundingVideo, depthVideo].forEach((videoEl) => {
      if (!videoEl) return;
      videoEl.muted = true;
      videoEl.defaultMuted = true;
      videoEl.autoplay = true;
      videoEl.loop = true;
      videoEl.playsInline = true;
      videoEl.setAttribute('playsinline', '');
      videoEl.addEventListener('loadeddata', () => safePlay(videoEl));
      videoEl.addEventListener('canplay', () => safePlay(videoEl));
    });
    syncTwoVideos(groundingVideo, depthVideo);
  });

  if (prevBtn) { prevBtn.addEventListener('click', () => goToSlide(currentIndex - 1)); prevBtn.disabled = true; }
  if (nextBtn) { nextBtn.addEventListener('click', () => goToSlide(currentIndex + 1)); nextBtn.disabled = slideEls.length <= 1; }

  let touchStartX = 0;
  sliderEl.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches ? e.changedTouches[0].screenX : e.screenX;
  }, { passive: true });
  sliderEl.addEventListener('touchend', (e) => {
    const touchEndX = e.changedTouches ? e.changedTouches[0].screenX : e.screenX;
    const diff = touchStartX - touchEndX;
    if (Math.abs(diff) > 50) {
      if (diff > 0) goToSlide(Math.min(currentIndex + 1, slideEls.length - 1));
      else goToSlide(Math.max(currentIndex - 1, 0));
    }
  }, { passive: true });

  goToSlide(0);
})();

// ── QA Carousel ──

function parseQA(raw, answer) {
  const sourceQuestion = typeof raw === 'string' ? raw : '';
  const sourceAnswer = typeof answer === 'string' ? answer : String(answer ?? '');
  const lines = sourceQuestion.split('\n');
  let questionLines = [];
  let options = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (/^[A-D][).]\s/.test(trimmed)) {
      options.push(trimmed);
    } else if (/^Reply with|^Return ONLY|^Output format|^Outputformat|^This is NOT/i.test(trimmed)) {
      continue;
    } else if (options.length === 0) {
      questionLines.push(trimmed);
    }
  }

  let questionText = questionLines.join(' ');
  if (questionText.length > 140) questionText = questionText.slice(0, 137) + '…';

  const answerLetter = sourceAnswer.trim().charAt(0).toUpperCase();

  return { questionText, options, answerLetter };
}

function buildCarouselCard(typeItem) {
  const examples = Array.isArray(typeItem && typeItem.examples) ? typeItem.examples : [];
  const example = examples[0];
  if (!example || typeof example !== 'object') return null;
  const exampleQuestion = typeof example.question === 'string' ? example.question : '';
  const exampleAnswer = typeof example.answer === 'string' ? example.answer : '';
  const images = Array.isArray(example.images)
    ? example.images.filter((img) => typeof img === 'string' && img.trim())
    : [];
  const { questionText, options, answerLetter } = parseQA(example.question, example.answer);

  const card = document.createElement('div');
  card.className = 'glass-card glass-blur rounded-2xl p-5 flex flex-col gap-4 snap-center shrink-0';
  card.style.width = '340px';

  const imgRow = document.createElement('div');
  imgRow.className = 'flex gap-2';
  images.slice(0, 2).forEach((imgPath, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'relative rounded-xl overflow-hidden flex-1 aspect-[4/3]';
    const img = document.createElement('img');
    img.src = IMAGE_PREFIX + imgPath;
    img.alt = 'View ' + (i === 0 ? 'A' : 'B');
    img.className = 'w-full h-full object-cover cursor-zoom-in';
    img.loading = 'lazy';
    img.addEventListener('click', () => openLightbox(IMAGE_PREFIX + imgPath));
    const badge = document.createElement('span');
    badge.className = 'absolute top-1.5 left-1.5 text-[10px] font-bold bg-black/60 backdrop-blur-sm text-white px-2 py-0.5 rounded-md';
    badge.textContent = i === 0 ? 'A' : 'B';
    wrap.appendChild(img);
    wrap.appendChild(badge);
    imgRow.appendChild(wrap);
  });
  if (!images.length) {
    const emptyState = document.createElement('div');
    emptyState.className = 'w-full rounded-xl border border-white/10 bg-white/[0.02] text-slate-500 text-xs px-3 py-6 text-center';
    emptyState.textContent = 'No preview image available';
    imgRow.appendChild(emptyState);
  }
  card.appendChild(imgRow);

  const body = document.createElement('div');
  body.className = 'space-y-3 flex-1 flex flex-col';

  const qDiv = document.createElement('div');
  qDiv.className = 'flex-1';
  const qLabel = document.createElement('span');
  qLabel.className = 'text-xs font-bold text-indigo-400 tracking-wider uppercase';
  qLabel.textContent = 'Question:';
  const qText = document.createElement('p');
  qText.className = 'text-white text-sm mt-1 leading-relaxed';
  qText.textContent = questionText;
  qDiv.appendChild(qLabel);
  qDiv.appendChild(qText);
  body.appendChild(qDiv);

  if (options.length > 0) {
    const optRow = document.createElement('div');
    optRow.className = 'flex flex-wrap gap-1.5';
    options.forEach((opt) => {
      const letter = opt.charAt(0).toUpperCase();
      const isCorrect = letter === answerLetter;
      const span = document.createElement('span');
      span.className = isCorrect
        ? 'text-[11px] px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-indigo-200 font-bold'
        : 'text-[11px] px-2.5 py-1 rounded-lg bg-white/[0.04] border border-white/[0.06] text-slate-400';
      const label = opt.replace(/^([A-D])[).]\s*/, '$1) ');
      span.textContent = isCorrect ? label + ' ✓' : label;
      optRow.appendChild(span);
    });
    body.appendChild(optRow);
  } else {
    const ansDiv = document.createElement('div');
    ansDiv.className = 'inline-flex items-center gap-2 bg-indigo-500/15 border border-indigo-400/30 rounded-lg px-3 py-1.5 self-start';
    const aLabel = document.createElement('span');
    aLabel.className = 'text-[10px] uppercase tracking-widest text-indigo-400';
    aLabel.textContent = 'Answer';
    const aVal = document.createElement('span');
    aVal.className = 'text-indigo-200 text-sm font-semibold';
    aVal.textContent = exampleAnswer || '-';
    ansDiv.appendChild(aLabel);
    ansDiv.appendChild(aVal);
    body.appendChild(ansDiv);
  }

  const typeDiv = document.createElement('div');
  typeDiv.className = 'flex items-center gap-2 pt-2 border-t border-white/5';
  const tLabel = document.createElement('span');
  tLabel.className = 'text-[10px] uppercase tracking-widest text-slate-500';
  tLabel.textContent = 'Type';
  const tVal = document.createElement('span');
  tVal.className = 'text-xs text-slate-300';
  tVal.textContent = (typeItem && typeItem.question_type) ? typeItem.question_type : 'Unknown';
  typeDiv.appendChild(tLabel);
  typeDiv.appendChild(tVal);
  body.appendChild(typeDiv);

  card.appendChild(body);
  return card;
}

function renderCarousel(data) {
  const carousel = document.getElementById('qa-carousel');
  const dotsContainer = document.getElementById('carousel-dots');
  const prevBtn = document.getElementById('carousel-prev');
  const nextBtn = document.getElementById('carousel-next');
  if (!carousel || !dotsContainer) return;

  const rawTypes = Array.isArray(data && data.types) ? data.types : [];
  const validTypes = rawTypes.filter((typeItem) => {
    return typeItem && typeof typeItem === 'object' && Array.isArray(typeItem.examples) && typeItem.examples.length > 0;
  });

  if (!validTypes.length) {
    const fallback = document.createElement('div');
    fallback.className = 'w-full text-center text-slate-400 py-8';
    fallback.textContent = 'No QA examples available.';
    carousel.appendChild(fallback);
    return;
  }

  validTypes.forEach((typeItem, i) => {
    const card = buildCarouselCard(typeItem);
    if (!card) return;
    carousel.appendChild(card);

    const dot = document.createElement('button');
    dot.className = 'w-2 h-2 rounded-full bg-white/20 transition-all cursor-pointer';
    dot.setAttribute('aria-label', typeItem.question_type || ('Type ' + (i + 1)));
    dot.addEventListener('click', () => {
      carousel.children[i].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    });
    dotsContainer.appendChild(dot);
  });

  const layoutState = { step: 360 };

  function updateLayoutStep() {
    const firstCard = carousel.querySelector('.glass-card');
    if (!firstCard) return;
    const styles = window.getComputedStyle(carousel);
    const gap = parseFloat(styles.columnGap || styles.gap || '20') || 20;
    layoutState.step = firstCard.getBoundingClientRect().width + gap;
  }

  function updateDots() {
    const scrollLeft = carousel.scrollLeft;
    const activeIndex = Math.round(scrollLeft / layoutState.step);
    dotsContainer.querySelectorAll('button').forEach((dot, i) => {
      dot.className = i === activeIndex
        ? 'w-6 h-2 rounded-full bg-indigo-400 transition-all cursor-pointer'
        : 'w-2 h-2 rounded-full bg-white/20 transition-all cursor-pointer';
    });
  }

  carousel.addEventListener('scroll', updateDots, { passive: true });
  updateLayoutStep();
  setTimeout(updateDots, 100);
  window.addEventListener('resize', updateLayoutStep, { passive: true });

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      carousel.scrollBy({ left: -layoutState.step, behavior: 'smooth' });
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      carousel.scrollBy({ left: layoutState.step, behavior: 'smooth' });
    });
  }
}

if (window.QA_DATA) {
  renderCarousel(window.QA_DATA);
} else {
  fetch(getQaDataUrl('qa_visualize/data.json'))
    .then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then((data) => renderCarousel(data))
    .catch((err) => {
      logWarn('qa carousel data load failed', err);
      const carousel = document.getElementById('qa-carousel');
      if (!carousel) return;
      const fallback = document.createElement('div');
      fallback.className = 'w-full text-center text-slate-400 py-8';
      fallback.textContent = 'Failed to load QA data.';
      carousel.appendChild(fallback);
    });
}

// ── Scroll Animations ──

function isElementInViewport(el) {
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight && rect.bottom > 0;
}

function revealVisibleElements() {
  document.querySelectorAll('.animate-on-scroll:not(.visible)').forEach((el) => {
    if (isElementInViewport(el)) el.classList.add('visible');
  });
}

function initScrollAnimations() {
  document.documentElement.classList.add('js-animations');
  revealVisibleElements();
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries) => { entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); } }); },
      { threshold: 0.05, rootMargin: '0px 0px 50px 0px' }
    );
    document.querySelectorAll('.animate-on-scroll:not(.visible)').forEach((el) => observer.observe(el));
    return;
  }
  let revealRaf = 0;
  const onScroll = () => {
    if (revealRaf) return;
    revealRaf = requestAnimationFrame(() => {
      revealRaf = 0;
      revealVisibleElements();
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  setTimeout(revealVisibleElements, 100);
  setTimeout(revealVisibleElements, 500);
}
initScrollAnimations();

// ── Visibility-driven video playback ──

function initVisibilityVideoPlayback() {
  const videos = Array.from(
    document.querySelectorAll('video[autoplay], video[data-autoplay-managed="true"]')
  ).filter((v) => v.id !== 'modalVideo');
  if (!videos.length) return;

  const inViewMap = new WeakMap();
  const managedVideos = new Set(videos);

  function shouldPlay(videoEl) {
    return !document.hidden && !!inViewMap.get(videoEl);
  }

  function applyPlayback(videoEl) {
    if (!managedVideos.has(videoEl)) return;
    if (shouldPlay(videoEl)) {
      const p = videoEl.play();
      if (p && typeof p.then === 'function') {
        p.catch((err) => logWarn('autoplay managed video play failed', err));
      }
    } else {
      videoEl.pause();
    }
  }

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          inViewMap.set(entry.target, entry.isIntersecting && entry.intersectionRatio > 0.2);
          applyPlayback(entry.target);
        });
      },
      { threshold: [0, 0.2, 0.5], rootMargin: '120px 0px 120px 0px' }
    );
    managedVideos.forEach((videoEl) => {
      inViewMap.set(videoEl, false);
      observer.observe(videoEl);
    });
  } else {
    managedVideos.forEach((videoEl) => inViewMap.set(videoEl, true));
  }

  document.addEventListener('visibilitychange', () => {
    managedVideos.forEach((videoEl) => applyPlayback(videoEl));
  });

  managedVideos.forEach((videoEl) => {
    videoEl.muted = true;
    videoEl.defaultMuted = true;
    videoEl.loop = true;
    videoEl.playsInline = true;
    videoEl.setAttribute('playsinline', '');
    applyPlayback(videoEl);
  });
}

initVisibilityVideoPlayback();

// ── Counter Animation ──

function animateCounter(el, target, duration = 2000) {
  let start = null;
  const step = (ts) => {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(eased * target).toLocaleString();
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = target.toLocaleString() + '+';
  };
  requestAnimationFrame(step);
}
document.querySelectorAll('[data-count]').forEach((el) => animateCounter(el, parseInt(el.dataset.count)));

// ── Scene Carousel Arrows (sync both rows) ──

function scrollAllBaselines(delta) {
  document.querySelectorAll('.scene-baselines').forEach((el) => {
    const firstCard = el.querySelector('.viewer-card');
    const cardWidth = firstCard ? firstCard.offsetWidth : 400;
    el.scrollBy({ left: delta * (cardWidth + 12), behavior: 'smooth' });
  });
}

document.querySelectorAll('.scene-prev').forEach((btn) => {
  btn.addEventListener('click', () => scrollAllBaselines(-1));
});
document.querySelectorAll('.scene-next').forEach((btn) => {
  btn.addEventListener('click', () => scrollAllBaselines(1));
});

// ── Mobile Menu ──

const menuToggle = document.getElementById('menu-toggle');
const menuClose = document.getElementById('menu-close');
const mobileMenu = document.getElementById('mobile-menu');
const menuBackdrop = document.getElementById('menu-backdrop');

function openMenu() {
  mobileMenu.classList.add('open');
  menuBackdrop.classList.remove('hidden');
  if (!isMenuOpen) {
    isMenuOpen = true;
    lockBodyScroll();
  }
}

function closeMenu() {
  mobileMenu.classList.remove('open');
  menuBackdrop.classList.add('hidden');
  if (isMenuOpen) {
    isMenuOpen = false;
    unlockBodyScroll();
  }
}

if (menuToggle) menuToggle.addEventListener('click', openMenu);
if (menuClose) menuClose.addEventListener('click', closeMenu);
if (menuBackdrop) menuBackdrop.addEventListener('click', closeMenu);
if (mobileMenu) mobileMenu.querySelectorAll('a').forEach((a) => a.addEventListener('click', closeMenu));
