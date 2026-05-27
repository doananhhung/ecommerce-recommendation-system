/* ==========================================================================
   PRODUCTION-GRADE JS LOGIC & CONTROL SYSTEM FOR INTERACTIVE SLIDES
   SUPPORTED KEYBOARD SHORTCUTS: NEXT (Space/->), PREV (<-), FULLSCREEN (P)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let currentSlide = 1;
    const totalSlides = 40;
    const slides = [];

    // Cache slides elements
    for (let i = 1; i <= totalSlides; i++) {
        const slideEl = document.getElementById(`slide-${i}`);
        if (slideEl) slides.push(slideEl);
    }

    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const currentSlideNum = document.getElementById('current-slide-num');
    const totalSlidesNum = document.getElementById('total-slides-num');
    const progressBar = document.getElementById('slide-progress-bar');
    const timelineProgress = document.getElementById('timeline-progress-bar');
    const timelineSteps = document.querySelectorAll('.timeline-step');

    // Cache speaker notes modal elements
    const notesModal = document.getElementById('notes-modal');
    const notesCloseBtn = document.getElementById('notes-close-btn');
    const notesModalBody = document.getElementById('notes-modal-body');

    if (totalSlidesNum) totalSlidesNum.textContent = totalSlides;

    function updateSlideUI() {
        // 1. Update counter
        if (currentSlideNum) currentSlideNum.textContent = currentSlide;

        // 2. Update navigation buttons state
        if (btnPrev) btnPrev.disabled = currentSlide === 1;
        if (btnNext) btnNext.disabled = currentSlide === totalSlides;

        // 3. Update main bottom progress bar
        if (progressBar) {
            const percent = ((currentSlide - 1) / (totalSlides - 1)) * 100;
            progressBar.style.width = `${percent}%`;
        }

        // 4. Update slide classes to trigger CSS transitions
        slides.forEach((slide, idx) => {
            const slideNum = idx + 1;
            slide.classList.remove('active', 'exit-left', 'enter-right');
            
            if (slideNum === currentSlide) {
                slide.classList.add('active');
            } else if (slideNum < currentSlide) {
                slide.classList.add('exit-left');
            } else {
                slide.classList.add('enter-right');
            }
        });

        // 5. Update interactive timeline progress (Slide 2 mapping)
        if (timelineProgress && timelineSteps.length > 0) {
            // Timeline steps mapping:
            // Step 1 (Phase 1): Slide 3 to 6
            // Step 2 (Phase 2): Slide 7 to 16
            // Step 3 (Phase 3): Slide 17 to 24 (Slide 17 is Architecture Workflow, Slides 18-24 are Recall Stage)
            // Step 4 (Phase 4): Slide 25 to 29
            // Step 5 (Phase 5): Slide 30 to 40
            
            timelineSteps.forEach((step, idx) => {
                step.classList.remove('active', 'completed');
                
                if (idx === 0) { // Phase 1 (Slide 3-6)
                    if (currentSlide >= 3 && currentSlide <= 6) step.classList.add('active');
                    else if (currentSlide > 6) step.classList.add('completed');
                } else if (idx === 1) { // Phase 2 (Slides 7-16)
                    if (currentSlide >= 7 && currentSlide <= 16) step.classList.add('active');
                    else if (currentSlide > 16) step.classList.add('completed');
                } else if (idx === 2) { // Phase 3 (Slides 17-24)
                    if (currentSlide >= 17 && currentSlide <= 24) step.classList.add('active');
                    else if (currentSlide > 24) step.classList.add('completed');
                } else if (idx === 3) { // Phase 4 (Slides 25-29)
                    if (currentSlide >= 25 && currentSlide <= 29) step.classList.add('active');
                    else if (currentSlide > 29) step.classList.add('completed');
                } else if (idx === 4) { // Phase 5 (Slides 30-40)
                    if (currentSlide >= 30) step.classList.add('active');
                }
            });

            // Calculate exact progress percentage on timeline
            let progressPercent = 0;
            if (currentSlide < 3) progressPercent = 0;
            else if (currentSlide >= 3 && currentSlide <= 6) {
                progressPercent = ((currentSlide - 3) / (6 - 3)) * 25;
            } else if (currentSlide >= 7 && currentSlide <= 16) {
                progressPercent = 25 + ((currentSlide - 7) / (16 - 7)) * 25;
            } else if (currentSlide >= 17 && currentSlide <= 24) {
                progressPercent = 50 + ((currentSlide - 17) / (24 - 17)) * 25;
            } else if (currentSlide >= 25 && currentSlide <= 29) {
                progressPercent = 75 + ((currentSlide - 25) / (29 - 25)) * 20;
            } else if (currentSlide >= 30) {
                progressPercent = 95 + ((currentSlide - 30) / (40 - 30)) * 5;
            }
            
            progressPercent = Math.min(100, Math.max(0, progressPercent));
            timelineProgress.style.width = `${progressPercent}%`;
        }

        // 6. Dynamic background glowing orbs transitions
        const orb1 = document.getElementById('orb-1');
        const orb2 = document.getElementById('orb-2');
        
        if (orb1 && orb2) {
            if (currentSlide === 1) {
                orb1.style.transform = 'translate(0px, 0px) scale(1)';
                orb2.style.transform = 'translate(0px, 0px) scale(1)';
            } else if (currentSlide === 4) {
                orb1.style.transform = 'translate(-250px, 120px) scale(1.1)';
                orb2.style.transform = 'translate(250px, -120px) scale(0.9)';
                // Re-trigger bar animations for Slide 4
                setTimeout(() => {
                    const fills = document.querySelectorAll('.chart-bar-fill');
                    fills.forEach(fill => {
                        const styleWidth = fill.style.width || fill.getAttribute('data-width');
                        if (styleWidth) {
                            if (!fill.getAttribute('data-width')) fill.setAttribute('data-width', styleWidth);
                            fill.style.width = '0%';
                            setTimeout(() => {
                                fill.style.width = styleWidth;
                            }, 60);
                        }
                    });
                }, 100);
            } else if (currentSlide === 10) {
                orb1.style.transform = 'translate(-120px, -180px) scale(0.85)';
                orb2.style.transform = 'translate(120px, 180px) scale(1.15)';
            } else if (currentSlide === 20) {
                orb1.style.transform = 'translate(120px, 220px) scale(1.1)';
                orb2.style.transform = 'translate(-120px, -220px) scale(0.9)';
            } else if (currentSlide === 24) {
                orb1.style.transform = 'translate(180px, -120px) scale(0.9)';
                orb2.style.transform = 'translate(-180px, 120px) scale(1.1)';
            } else if (currentSlide === 26) {
                orb1.style.transform = 'translate(-120px, 180px) scale(1.15)';
                orb2.style.transform = 'translate(120px, -180px) scale(0.85)';
            } else if (currentSlide === 39) {
                orb1.style.transform = 'translate(0px, 0px) scale(1)';
                orb2.style.transform = 'translate(0px, 0px) scale(1)';
                // Re-trigger latency bar animations for Slide 39
                setTimeout(() => {
                    const latencyFills = document.querySelectorAll('.latency-bar-fill');
                    latencyFills.forEach(fill => {
                        const styleWidth = fill.style.width || fill.getAttribute('data-width');
                        if (styleWidth) {
                            if (!fill.getAttribute('data-width')) fill.setAttribute('data-width', styleWidth);
                            fill.style.width = '0%';
                            setTimeout(() => {
                                fill.style.width = styleWidth;
                            }, 60);
                        }
                    });
                }, 100);
            } else {
                orb1.style.transform = `translate(${Math.sin(currentSlide) * 60}px, ${Math.cos(currentSlide) * 60}px) scale(1)`;
                orb2.style.transform = `translate(${Math.cos(currentSlide) * -60}px, ${Math.sin(currentSlide) * -60}px) scale(1)`;
            }
        }

        // Update speaker notes content automatically
        updateSpeakerNotes();
    }

    function changeSlide(direction) {
        const nextSlide = currentSlide + direction;
        if (nextSlide >= 1 && nextSlide <= totalSlides) {
            currentSlide = nextSlide;
            updateSlideUI();
        }
    }

    function goToSlide(slideNum) {
        if (slideNum >= 1 && slideNum <= totalSlides) {
            currentSlide = slideNum;
            updateSlideUI();
        }
    }

    // --- SPEAKER NOTES CONTROLLER ---
    function updateSpeakerNotes() {
        if (notesModalBody && typeof SPEAKER_NOTES !== 'undefined') {
            const notes = SPEAKER_NOTES[currentSlide];
            notesModalBody.innerHTML = notes || `<strong>Slide ${currentSlide}</strong><br><br>Không có kịch bản thuyết trình cho slide này.`;
        }
    }

    function toggleSpeakerNotes() {
        if (notesModal) {
            notesModal.classList.toggle('active');
        }
    }

    // Close notes modal on close button click
    if (notesCloseBtn) {
        notesCloseBtn.addEventListener('click', toggleSpeakerNotes);
    }

    // Close notes modal when clicking outside the content block
    if (notesModal) {
        notesModal.addEventListener('click', (e) => {
            if (e.target === notesModal) {
                toggleSpeakerNotes();
            }
        });
    }

    // --- FULLSCREEN CONTROLLER ---
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            // Promotes whole document to standard fullscreen mode
            document.documentElement.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable full-screen mode: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

    // Listen to fullscreen changes to align classes for styles
    document.addEventListener('fullscreenchange', () => {
        if (document.fullscreenElement) {
            document.body.classList.add('fullscreen-active');
        } else {
            document.body.classList.remove('fullscreen-active');
        }
    });

    // Expose control functions globally so onclick handlers work
    window.changeSlide = changeSlide;
    window.goToSlide = goToSlide;
    window.toggleFullscreen = toggleFullscreen;
    window.toggleSpeakerNotes = toggleSpeakerNotes;

    // --- KEYBOARD CONTROLS ---
    document.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();
        
        if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
            e.preventDefault();
            changeSlide(1);
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
            e.preventDefault();
            changeSlide(-1);
        } else if (e.key === 'Home') {
            e.preventDefault();
            goToSlide(1);
        } else if (e.key === 'End') {
            e.preventDefault();
            goToSlide(totalSlides);
        } else if (key === 'p') {
            e.preventDefault();
            toggleFullscreen();
        } else if (key === 's') {
            e.preventDefault();
            toggleSpeakerNotes();
        }
    });

    // Initialize UI on startup
    updateSlideUI();
});
