document.addEventListener('DOMContentLoaded', () => {
    let currentSlide = 1;
    const totalSlides = 22; // Updated to 22 slides
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
    const notesModal = document.getElementById('notes-modal');
    const notesCloseBtn = document.getElementById('notes-close-btn');
    const notesModalBody = document.getElementById('notes-modal-body');

    if (totalSlidesNum) totalSlidesNum.textContent = totalSlides;

    function updateSlideUI() {
        if (currentSlideNum) currentSlideNum.textContent = currentSlide;
        if (btnPrev) btnPrev.disabled = currentSlide === 1;
        if (btnNext) btnNext.disabled = currentSlide === totalSlides;
        
        if (progressBar) {
            const percent = ((currentSlide - 1) / (totalSlides - 1)) * 100;
            progressBar.style.width = `${percent}%`;
        }

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

        // Dynamic background glowing orbs transitions
        const orb1 = document.getElementById('orb-1');
        const orb2 = document.getElementById('orb-2');
        if (orb1 && orb2) {
            orb1.style.transform = `translate(${Math.sin(currentSlide) * 60}px, ${Math.cos(currentSlide) * 60}px) scale(1)`;
            orb2.style.transform = `translate(${Math.cos(currentSlide) * -60}px, ${Math.sin(currentSlide) * -60}px) scale(1)`;
        }

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
            notesModalBody.innerHTML = notes || `<strong>Slide ${currentSlide}</strong><br><br>Không có kịch bản thuyết trình.`;
        }
    }

    function toggleSpeakerNotes() {
        if (notesModal) {
            notesModal.classList.toggle('active');
        }
    }

    if (notesCloseBtn) notesCloseBtn.addEventListener('click', toggleSpeakerNotes);
    if (notesModal) {
        notesModal.addEventListener('click', (e) => {
            if (e.target === notesModal) toggleSpeakerNotes();
        });
    }

    // --- FULLSCREEN CONTROLLER ---
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable full-screen mode: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

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
