// ==UserScript==
// @name         HSBG Coach — HSReplay capture
// @namespace    hsbg-coach
// @version      1.0
// @description  Records the BG stats payloads hsreplay.net fetches while you browse, and downloads them as one bundle for `python -m hsbg_coach import-hsreplay`.
// @match        https://hsreplay.net/*
// @run-at       document-start
// @grant        none
// ==/UserScript==

// Cloudflare-proof capture: this runs inside your NORMAL browser (install the
// Tampermonkey extension, then open this file's raw URL or paste it into a new
// Tampermonkey script). No automation for the site to detect.
//
// Use: browse the Battlegrounds section (minions / comps / heroes / trinkets /
// dark gifts) flipping filters — a badge bottom-right counts captures. Click
// the badge to download hsreplay_bundle.json, then on your machine:
//     python -m hsbg_coach import-hsreplay ~/Downloads/hsreplay_bundle.json
(function () {
  "use strict";
  const INTERESTING = /hsreplay\.net\/(analytics|api)\//i;
  const captures = [];

  function record(url, method, postData, body) {
    if (!INTERESTING.test(url)) return;
    captures.push({
      url: String(url),
      method: method || "GET",
      post_data: postData ? String(postData) : null,
      fetched_at: new Date().toISOString().slice(0, 19),
      body: body,
    });
    badge.textContent = `HSBG: ${captures.length} captured — click to download`;
    badge.style.display = "block";
  }

  // fetch hook
  const origFetch = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const method = (init && init.method) || (input && input.method) || "GET";
    const postData = init && typeof init.body === "string" ? init.body : null;
    return origFetch.apply(this, arguments).then((resp) => {
      try {
        if (INTERESTING.test(url) &&
            (resp.headers.get("content-type") || "").includes("json")) {
          resp.clone().json().then(
            (body) => record(resp.url || url, method, postData, body),
            () => {});
        }
      } catch (e) {}
      return resp;
    });
  };

  // XHR hook
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url) {
    this.__hsbg = { method: method, url: url };
    return origOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function (body) {
    const meta = this.__hsbg;
    if (meta && INTERESTING.test(meta.url)) {
      this.addEventListener("load", () => {
        try {
          const ctype = this.getResponseHeader("content-type") || "";
          if (!ctype.includes("json")) return;
          record(meta.url, meta.method,
                 typeof body === "string" ? body : null,
                 JSON.parse(this.responseText));
        } catch (e) {}
      });
    }
    return origSend.apply(this, arguments);
  };

  // badge + download
  const badge = document.createElement("div");
  badge.style.cssText =
    "position:fixed;bottom:14px;right:14px;z-index:2147483647;display:none;" +
    "background:#1d3657;color:#fff;padding:8px 14px;border-radius:8px;" +
    "font:13px/1.4 sans-serif;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.4)";
  badge.title = "Download the captured HSReplay payloads as a bundle";
  badge.addEventListener("click", () => {
    const blob = new Blob(
      [JSON.stringify({ captures: captures })],
      { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "hsreplay_bundle.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  (document.body || document.documentElement).appendChild(badge);
  document.addEventListener("DOMContentLoaded", () =>
    document.body.appendChild(badge));
})();
