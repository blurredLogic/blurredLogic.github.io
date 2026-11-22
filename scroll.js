(() => {
  const root = document.documentElement;
  const preference = matchMedia('(prefers-reduced-motion: reduce)');
  const hero = document.querySelector('.hero');
  const art = hero.querySelector('.hero-art');
  const sections = [...document.querySelectorAll('main > section:not(.hero)')];
  const clamp = value => Math.max(0, Math.min(1, value));
  // Begin once the element's natural top is well inside the viewport.
  const revealStart = 0.74;
  const detailGroups = [
    ['.intro .doodle', 'doodle'],
    ['.mini-facts > div', 'fact'],
    ['.experience-gear', 'gear'],
    ['.impact-list > li', 'impact'],
    ['.toolkit-list > div', 'tool']
  ];
  let frame = 0;

  const groups = sections.map(section => {
    section.dataset.scrollSection = '';
    const elements = [...section.querySelectorAll(
      '.intro-grid > div:last-child, .section-heading, .experience-card, .project-card, ' +
      '.toolkit-grid > div, .education-section h2, .education-list > div, .contact-inner > h2'
    )];
    elements.forEach(element => { element.dataset.scrollContent = ''; });
    const details = detailGroups.flatMap(([selector, effect]) =>
      [...section.querySelectorAll(selector)].map((element, index) => {
        element.dataset.scrollDetail = effect;
        return { element, index };
      })
    );
    return { section, elements, details };
  });

  function layoutTop(element) {
    let top = 0;
    for (let node = element; node; node = node.offsetParent) top += node.offsetTop;
    return top;
  }

  function paint() {
    frame = 0;
    const height = root.clientHeight;
    const scrollY = window.scrollY;
    const heroProgress = clamp(scrollY / (hero.offsetHeight * 0.75));
    const reading = clamp(scrollY / Math.max(1, root.scrollHeight - height));

    // Read the untransformed layout before updating any styles.
    const positions = groups.map(({ section, elements, details }) => ({
      section,
      top: section.getBoundingClientRect().top,
      entries: elements.map(element => ({ element, top: layoutTop(element) - scrollY })),
      details: details.map(({ element, index }) => ({
        element,
        progress: clamp((height * (revealStart - index * 0.045) - (layoutTop(element) - scrollY)) / (height * 0.36))
      }))
    }));

    root.style.setProperty('--reading', reading);
    art.style.setProperty('--hero-offset', `${-104 * heroProgress}px`);
    art.style.setProperty('--hero-turn', `${24 * heroProgress}deg`);
    art.style.setProperty('--orbit-turn', `${-29 * heroProgress}deg`);

    positions.forEach(({ section, top, entries, details }) => {
      const boundary = clamp((height * (revealStart + 0.06) - top) / (height * 0.42));
      section.style.setProperty('--section-progress', boundary);
      let projectRowTop;
      let projectColumn = 0;

      entries.forEach(({ element, top: entryTop }) => {
        const isProject = element.classList.contains('project-card');
        let start = revealStart;
        let distance = 0.34;

        if (isProject) {
          // Stagger cards within each row. Stacked cards follow their own viewport entry.
          projectColumn = entryTop === projectRowTop ? projectColumn + 1 : 0;
          projectRowTop = entryTop;
          distance = 0.16;
          // Leave 0.09 viewport heights after a card settles before the next begins.
          start = revealStart - projectColumn * 0.25;
        }

        const progress = clamp((height * start - entryTop) / (height * distance));
        const remaining = (1 - progress) ** 1.5;
        element.style.setProperty('--entry-offset', `${(isProject ? 64 : 44) * remaining}px`);
        element.style.setProperty('--entry-opacity', 1 - (isProject ? 0.88 : 0.58) * remaining);
      });
      details.forEach(({ element, progress }) => {
        element.style.setProperty('--detail-progress', 1 - (1 - progress) ** 1.5);
      });
    });
  }

  function schedule() {
    if (!frame && !preference.matches) frame = requestAnimationFrame(paint);
  }

  function syncPreference() {
    root.classList.toggle('scroll-active', !preference.matches);
    schedule();
  }

  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule);
  addEventListener('pageshow', schedule);
  preference.addEventListener('change', syncPreference);
  syncPreference();
})();
