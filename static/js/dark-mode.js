// Dark mode functionality
function initDarkMode() {
    const themeToggle = document.querySelector('.theme-toggle');
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Check for saved theme preference or use system preference
    const currentTheme = localStorage.getItem('theme') || 
                        (prefersDarkScheme.matches ? 'dark' : 'light');
    
    // Set initial theme
    if (currentTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeToggle.classList.add('dark');
    }

    // Theme toggle click handler
    themeToggle.addEventListener('click', function() {
        let theme = 'light';
        
        if (!this.classList.contains('dark')) {
            theme = 'dark';
            this.classList.add('dark');
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            this.classList.remove('dark');
            document.documentElement.removeAttribute('data-theme');
        }
        
        localStorage.setItem('theme', theme);
    });

    // Listen for system theme changes
    prefersDarkScheme.addEventListener('change', function(e) {
        if (!localStorage.getItem('theme')) {
            if (e.matches) {
                document.documentElement.setAttribute('data-theme', 'dark');
                themeToggle.classList.add('dark');
            } else {
                document.documentElement.removeAttribute('data-theme');
                themeToggle.classList.remove('dark');
            }
        }
    });
}

// Initialize dark mode when DOM is loaded
document.addEventListener('DOMContentLoaded', initDarkMode);
