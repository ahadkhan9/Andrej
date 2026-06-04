document.addEventListener("DOMContentLoaded", () => {
    // 1. Theme Management (Default: Dark)
    const initTheme = () => {
        const savedTheme = localStorage.getItem("theme") || "dark";
        document.documentElement.setAttribute("data-theme", savedTheme);
        updateThemeToggleIcon(savedTheme);
    };

    const toggleTheme = () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
        updateThemeToggleIcon(newTheme);
    };

    const updateThemeToggleIcon = (theme) => {
        const toggleBtn = document.getElementById("theme-toggle");
        if (!toggleBtn) return;
        toggleBtn.innerHTML = theme === "dark" 
            ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
            : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
    };

    const toggleBtn = document.getElementById("theme-toggle");
    if (toggleBtn) {
        toggleBtn.addEventListener("click", toggleTheme);
    }
    initTheme();

    // 2. Dynamic Table of Contents (TOC) Builder
    const buildTOC = () => {
        const tocList = document.getElementById("toc-list");
        if (!tocList) return;

        const headings = document.querySelectorAll(".content-wrapper h2, .content-wrapper h3");
        if (headings.length === 0) {
            const tocSidebar = document.querySelector(".toc-sidebar");
            if (tocSidebar) tocSidebar.style.display = "none";
            return;
        }

        headings.forEach((heading, index) => {
            // Ensure heading has an ID for anchoring
            if (!heading.id) {
                heading.id = heading.textContent.toLowerCase()
                    .replace(/[^a-z0-9]+/g, "-")
                    .replace(/(^-|-$)/g, "");
            }

            const listItem = document.createElement("li");
            const link = document.createElement("a");
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent;
            link.className = "toc-link";
            if (heading.tagName === "H3") {
                link.classList.add("h3");
            }

            listItem.appendChild(link);
            tocList.appendChild(listItem);
        });
    };
    buildTOC();

    // 3. Syntax Highlighting Headers & Copy-to-Clipboard
    const configureCodeBlocks = () => {
        const codeBlocks = document.querySelectorAll("pre");
        codeBlocks.forEach((block) => {
            // Find parent language or attribute if possible
            const codeEl = block.querySelector("code");
            const langClass = codeEl ? codeEl.className : "";
            const lang = langClass.replace("language-", "") || "code";

            // Create header bar
            const header = document.createElement("div");
            header.className = "code-header";

            const langLabel = document.createElement("span");
            langLabel.textContent = lang.toUpperCase();
            header.appendChild(langLabel);

            const copyButton = document.createElement("button");
            copyButton.className = "copy-btn";
            copyButton.textContent = "Copy";
            copyButton.addEventListener("click", () => {
                const codeText = codeEl ? codeEl.innerText : block.innerText;
                navigator.clipboard.writeText(codeText).then(() => {
                    copyButton.textContent = "Copied!";
                    setTimeout(() => {
                        copyButton.textContent = "Copy";
                    }, 2000);
                }).catch(() => {
                    copyButton.textContent = "Failed";
                });
            });
            header.appendChild(copyButton);

            // Insert header right before block
            block.parentNode.insertBefore(header, block);
        });
    };
    configureCodeBlocks();

    // 4. Highlight Active Navigation Section
    const highlightActiveNav = () => {
        const currentFile = window.location.pathname.split("/").pop() || "index.html";
        const navLinks = document.querySelectorAll(".sidebar-nav-link");
        navLinks.forEach((link) => {
            const linkFile = link.getAttribute("href");
            if (linkFile === currentFile) {
                link.classList.add("active");
            } else {
                link.classList.remove("active");
            }
        });
    };
    highlightActiveNav();
});
