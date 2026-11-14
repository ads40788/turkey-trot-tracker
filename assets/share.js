/*
 * "Share to Instagram Story" button.
 *
 * On phones this opens the native share sheet with the story-sized card
 * attached, so the user taps Instagram → Story with no download step.
 *
 * Constraints this works around:
 *  - navigator.share() must run inside the user's tap. Safari is strict about
 *    this, so the image is fetched *before* the tap (prefetched whenever the
 *    selected runners change) and the click handler does no awaiting.
 *    That's also why this is a plain DOM listener rather than a Dash callback.
 *  - Websites can't pre-fill Instagram's link sticker (only approved native
 *    apps can), so the link is copied to the clipboard for the user to paste
 *    into a link sticker. The URL is also printed on the image.
 *
 * Dash serves every .js file in assets/ automatically.
 */
(function () {
    "use strict";

    var cache = { href: null, file: null };

    function storyHref() {
        var a = document.getElementById("download-story");
        return a ? a.getAttribute("href") : null;
    }

    function shareUrlFrom(href) {
        var u = new URL(href, location.origin);
        u.searchParams.delete("size");
        u.searchParams.delete("download");
        var q = u.searchParams.toString();
        return location.origin + "/" + (q ? "?" + q : "");
    }

    function fileNameFrom(href) {
        var u = new URL(href, location.origin);
        var parts = [u.searchParams.get("runner"), u.searchParams.get("vs")].filter(Boolean);
        var base = parts.length ? parts.join("-vs-") : "buffalo-turkey-trot";
        return base.replace(/[^a-z0-9]+/gi, "-").toLowerCase() + "-story.png";
    }

    function setStatus(msg) {
        var el = document.getElementById("ig-status");
        if (el) { el.textContent = msg; }
    }

    // Fetch the story image whenever the selection (and so the href) changes.
    function prefetch() {
        var href = storyHref();
        if (!href || href === cache.href) { return; }
        cache = { href: href, file: null };
        var imgUrl = href.replace(/&?download=1/, "");
        fetch(imgUrl)
            .then(function (r) { return r.ok ? r.blob() : Promise.reject(r.status); })
            .then(function (blob) {
                if (cache.href === href) {
                    cache.file = new File([blob], fileNameFrom(href), { type: "image/png" });
                }
            })
            .catch(function () { /* retried on the next selection change or tap */ });
    }
    setInterval(prefetch, 800);

    function copyLink(url) {
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(url).catch(function () {});
                return true;
            }
        } catch (e) { /* ignore */ }
        return false;
    }

    function downloadFallback(href, copied) {
        var a = document.createElement("a");
        a.href = href;
        a.download = fileNameFrom(href);
        document.body.appendChild(a);
        a.click();
        a.remove();
        setStatus(
            "Story image downloaded" + (copied ? " and link copied" : "") + ". " +
            "Instagram Stories can only be posted from the phone app — open this page " +
            "on your phone for one-tap sharing."
        );
    }

    document.addEventListener("click", function (e) {
        var btn = e.target.closest && e.target.closest("#ig-story-btn");
        if (!btn) { return; }
        e.preventDefault();

        var href = storyHref();
        if (!href) { return; }
        var url = shareUrlFrom(href);

        // Both calls below happen synchronously inside the tap.
        var copied = copyLink(url);
        var file = cache.href === href ? cache.file : null;
        // Desktop OSes also support file sharing, but Instagram isn't a share
        // target there, so only phones/tablets get the share sheet.
        // (iPadOS reports itself as a Mac, hence the touch-points check.)
        var isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
            (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));
        var canShareFile = !!(isMobile && file && navigator.canShare &&
                              navigator.canShare({ files: [file] }));

        if (canShareFile) {
            setStatus(copied
                ? "Link copied. In Instagram, choose Story, then add a link sticker and paste."
                : "In Instagram, choose Story, then add a link sticker with: " + url);
            navigator.share({ files: [file], title: "Buffalo Turkey Trot", text: url })
                .catch(function (err) {
                    if (err && err.name === "AbortError") { return; }  // user closed the sheet
                    downloadFallback(href, copied);
                });
            return;
        }

        if (isMobile && !file && navigator.canShare) {
            // Phone with sharing support but the image hasn't arrived yet.
            prefetch();
            setStatus("Getting the image ready… tap again in a second.");
            return;
        }

        // Desktop or a browser without file sharing.
        downloadFallback(href, copied);
    });
})();
