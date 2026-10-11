import { app } from "/scripts/app.js";

const BADGE_RADIUS = 7;
const BADGE_MARGIN = 16;
const TITLE_CENTER_Y = -15;

function badgeCenter(nodeWidth) {
    return [nodeWidth - BADGE_MARGIN, TITLE_CENTER_Y];
}

function showBadge(nodeWidth, collapsed) {
    return !collapsed && nodeWidth >= 120;
}

function hitsBadge(pos, nodeWidth) {
    const [cx, cy] = badgeCenter(nodeWidth);
    return (
        pos[0] >= cx - 10 && pos[0] <= cx + 10 &&
        pos[1] >= cy - 11 && pos[1] <= cy + 11
    );
}

function isMacOffloadNode(node) {
    const comfyClass = node?.comfyClass || "";
    return comfyClass.startsWith("MACOFFLOAD_");
}

function chainCallback(proto, name, fn) {
    const prior = proto[name];
    proto[name] = function (...args) {
        const result = prior?.apply(this, args);
        fn.apply(this, args);
        return result;
    };
}

function chainHandler(proto, name, fn) {
    const prior = proto[name];
    proto[name] = function (...args) {
        const result = prior?.apply(this, args);
        const handled = fn.apply(this, args);
        return result || handled;
    };
}

function openHelpDialog(node) {
    const nodeData = node.constructor?.nodeData || {};
    const title = node.title || nodeData.display_name || "About this node";
    const description = (
        nodeData.description ||
        "Offloads Text Encoder forward passes to a remote macOS server running on Apple Silicon MPS."
    );

    const container = document.createElement("div");
    container.style.maxWidth = "520px";
    container.style.lineHeight = "1.5";
    container.style.fontSize = "13px";
    container.style.color = "#ccc";
    container.style.fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

    // Title Header
    const h = document.createElement("h3");
    h.textContent = title;
    h.style.marginTop = "0";
    h.style.marginBottom = "10px";
    h.style.color = "#fff";
    h.style.borderBottom = "1px solid rgba(255,255,255,0.15)";
    h.style.paddingBottom = "8px";
    container.appendChild(h);

    // Description text
    const descP = document.createElement("p");
    descP.textContent = description;
    descP.style.marginBottom = "14px";
    descP.style.color = "#ddd";
    container.appendChild(descP);

    // Inputs & Parameters
    const inputs = [];
    const inputSpec = nodeData.input || {};
    for (const group of ["required", "optional"]) {
        for (const [name, spec] of Object.entries(inputSpec[group] || {})) {
            const options = Array.isArray(spec) ? spec[1] : undefined;
            const tooltip = options?.tooltip;
            inputs.push({
                name,
                type: Array.isArray(spec) ? spec[0] : "ANY",
                tooltip,
                group,
            });
        }
    }

    if (inputs.length > 0) {
        const inHead = document.createElement("h4");
        inHead.textContent = "Parameters & Inputs";
        inHead.style.margin = "12px 0 6px 0";
        inHead.style.color = "#61afef";
        container.appendChild(inHead);

        const list = document.createElement("div");
        list.style.display = "flex";
        list.style.flexDirection = "column";
        list.style.gap = "6px";
        list.style.marginBottom = "14px";

        for (const item of inputs) {
            const row = document.createElement("div");
            row.style.background = "rgba(255,255,255,0.05)";
            row.style.padding = "6px 8px";
            row.style.borderRadius = "4px";

            const nameSpan = document.createElement("strong");
            nameSpan.textContent = item.name;
            nameSpan.style.color = "#98c379";

            const typeSpan = document.createElement("span");
            const typeText = Array.isArray(item.type) ? "choice" : item.type;
            typeSpan.textContent = ` (${typeText}${item.group === "optional" ? ", optional" : ""})`;
            typeSpan.style.opacity = "0.7";
            typeSpan.style.fontSize = "11px";
            typeSpan.style.marginLeft = "4px";

            row.appendChild(nameSpan);
            row.appendChild(typeSpan);

            if (item.tooltip) {
                const tipDiv = document.createElement("div");
                tipDiv.textContent = item.tooltip;
                tipDiv.style.fontSize = "12px";
                tipDiv.style.opacity = "0.85";
                tipDiv.style.marginTop = "2px";
                row.appendChild(tipDiv);
            }
            list.appendChild(row);
        }
        container.appendChild(list);
    }

    // Mac Server Quick Verification
    const srvHead = document.createElement("h4");
    srvHead.textContent = "Mac Server Quick Verification";
    srvHead.style.margin = "12px 0 6px 0";
    srvHead.style.color = "#e5c07b";
    container.appendChild(srvHead);

    const srvP = document.createElement("div");
    srvP.style.background = "rgba(0,0,0,0.25)";
    srvP.style.padding = "8px 10px";
    srvP.style.borderRadius = "4px";
    srvP.style.fontSize = "12px";
    srvP.innerHTML = `
        <div>- <strong>Start server on Mac:</strong> <code>python server.py --port 8288</code></div>
        <div style="margin-top: 4px;">- <strong>Probe health:</strong> <code>http://&lt;mac-ip&gt;:8288/health</code></div>
        <div style="margin-top: 4px;">- <strong>Network check:</strong> Ensure Mac and Windows PC are on the same LAN subnet.</div>
    `;
    container.appendChild(srvP);

    if (app.ui?.dialog?.show) {
        app.ui.dialog.show(container);
    } else {
        // Fallback simple modal overlay
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.width = "100vw";
        overlay.style.height = "100vh";
        overlay.style.backgroundColor = "rgba(0,0,0,0.7)";
        overlay.style.display = "flex";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
        overlay.style.zIndex = "99999";

        const box = document.createElement("div");
        box.style.background = "#202020";
        box.style.padding = "20px";
        box.style.borderRadius = "8px";
        box.style.maxWidth = "550px";
        box.style.boxShadow = "0 4px 20px rgba(0,0,0,0.6)";

        const closeBtn = document.createElement("button");
        closeBtn.textContent = "Close";
        closeBtn.style.marginTop = "14px";
        closeBtn.style.padding = "6px 16px";
        closeBtn.style.background = "#444";
        closeBtn.style.color = "#fff";
        closeBtn.style.border = "none";
        closeBtn.style.borderRadius = "4px";
        closeBtn.style.cursor = "pointer";
        closeBtn.onclick = () => document.body.removeChild(overlay);

        box.appendChild(container);
        box.appendChild(closeBtn);
        overlay.appendChild(box);
        document.body.appendChild(overlay);
    }
}

function installHelpBadge(node) {
    chainCallback(node, "onDrawForeground", function (ctx) {
        if (!showBadge(this.size?.[0] ?? 0, this.flags?.collapsed)) return;
        const [x, y] = badgeCenter(this.size[0]);
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, BADGE_RADIUS, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(155, 162, 170, 0.6)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = "rgba(155, 162, 170, 0.85)";
        ctx.font = "bold 9px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("?", x, y + 0.5);
        ctx.restore();
    });

    chainHandler(node, "onMouseDown", function (event, pos) {
        if (!showBadge(this.size?.[0] ?? 0, this.flags?.collapsed)) return false;
        if (!pos || !hitsBadge(pos, this.size[0])) return false;
        openHelpDialog(this);
        return true;
    });
}

function installModelFetchButton(node) {
    node.addWidget("button", "🔄 Fetch Models from Server", null, async () => {
        const serverUrlWidget = node.widgets?.find(w => w.name === "server_url");
        const modelNameWidget = node.widgets?.find(w => w.name === "model_name");
        if (!modelNameWidget) return;

        const serverUrl = (serverUrlWidget?.value || "http://127.0.0.1:8288").trim().replace(/\/+$/, "");

        try {
            let data = null;
            let proxyErrorDetail = null;

            // 1. First try ComfyUI backend proxy (immune to browser CORS & LAN restrictions)
            try {
                const proxyUrl = `/macoffload/models?server_url=${encodeURIComponent(serverUrl)}`;
                const proxyRes = await fetch(proxyUrl);
                if (proxyRes.ok) {
                    data = await proxyRes.json();
                } else {
                    const errBody = await proxyRes.json().catch(() => null);
                    if (errBody?.error) {
                        proxyErrorDetail = errBody.error;
                    } else if (proxyRes.status !== 404) {
                        proxyErrorDetail = `Proxy HTTP ${proxyRes.status}: ${proxyRes.statusText}`;
                    }
                }
            } catch (err) {
                // ComfyUI backend proxy request failure
                proxyErrorDetail = err?.message || null;
            }

            // 2. Direct browser fetch fallback (if backend proxy is 404 or unconfigured)
            if (!data) {
                // If the proxy reached the target and reported a concrete connection refusal / timeout, surface it directly
                if (proxyErrorDetail && !proxyErrorDetail.includes("404")) {
                    throw new Error(proxyErrorDetail);
                }

                try {
                    const res = await fetch(`${serverUrl}/v1/models`);
                    if (!res.ok) {
                        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
                    }
                    data = await res.json();
                } catch (directErr) {
                    const detail = proxyErrorDetail || directErr.message;
                    throw new Error(detail);
                }
            }

            if (data.error) {
                throw new Error(data.error);
            }

            const allModels = data.models || [];
            if (!allModels.length) {
                alert(`Connected to ${serverUrl}, but no models were found in the server's models directory.`);
                return;
            }

            const isFlux = node.comfyClass?.includes("CLIPTextEncode");
            const filtered = allModels.filter(m => {
                const type = data.model_types?.[m];
                return type ? (isFlux ? type === "flux_klein" : type === "qwen_image") : true;
            });
            const targetList = filtered.length > 0 ? filtered : allModels;

            const currentValue = modelNameWidget.value;
            modelNameWidget.options = modelNameWidget.options || {};
            modelNameWidget.options.values = targetList;

            // Retain existing model name if present in new list (preserves workflow memory)
            if (targetList.includes(currentValue)) {
                modelNameWidget.value = currentValue;
            } else {
                modelNameWidget.value = targetList[0];
            }

            node.setDirtyCanvas(true, true);

            if (app.extensionManager?.toast?.add) {
                app.extensionManager.toast.add({
                    severity: "success",
                    summary: "MacOffload",
                    detail: `Fetched ${targetList.length} models from ${serverUrl}!`,
                    life: 3000,
                });
            }
        } catch (err) {
            let extraHint = "";
            if (serverUrl.includes("127.0.0.1") || serverUrl.includes("localhost")) {
                extraHint = "\n\n💡 Tip: Target is currently set to 127.0.0.1 (local). If te-server is running on a remote Mac, please change 'server_url' to your Mac's LAN address (e.g. http://<mac-ip>:8288).";
            }
            alert(
                `Failed to fetch models from ${serverUrl}:\n${err.message}${extraHint}`
            );
        }
    });
}

app.registerExtension({
    name: "MacOffload.HelpBadge",
    nodeCreated(node) {
        if (!isMacOffloadNode(node)) return;
        installHelpBadge(node);
        installModelFetchButton(node);
    },
    getNodeMenuItems(node) {
        if (!isMacOffloadNode(node)) return [];
        return [
            {
                content: "About this node ❓",
                callback: () => openHelpDialog(node),
            },
        ];
    },
});
