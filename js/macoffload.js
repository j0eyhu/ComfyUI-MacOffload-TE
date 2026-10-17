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

function getNodeClass(node) {
    return node?.comfyClass || node?.type || node?.constructor?.comfyClass || node?.constructor?.name || "";
}

function isMacOffloadNode(node) {
    return getNodeClass(node).startsWith("MACOFFLOAD_");
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

async function fetchCurrentServerUrl() {
    try {
        const res = await fetch("/macoffload/config");
        if (res.ok) {
            const data = await res.json();
            return data.server_url || "http://127.0.0.1:8288";
        }
    } catch (e) {
        console.warn("[MacOffload] Failed to read current config:", e);
    }
    return "http://127.0.0.1:8288";
}

async function saveServerUrl(url) {
    const clean = (url || "").trim().replace(/\/+$/, "");
    if (!clean) return;
    const res = await fetch("/macoffload/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ server_url: clean }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.server_url;
}

async function promptConfigureServerUrl() {
    const currentUrl = await fetchCurrentServerUrl();
    const newUrl = prompt(
        "Enter remote macOS server address (saved locally on disk, not in workflow):",
        currentUrl
    );
    if (newUrl === null) return; // User cancelled
    const cleanUrl = newUrl.trim();
    if (!cleanUrl) return;

    try {
        const saved = await saveServerUrl(cleanUrl);
        if (app.extensionManager?.toast?.add) {
            app.extensionManager.toast.add({
                severity: "success",
                summary: "MacOffload",
                detail: `Server address saved locally: ${saved}`,
                life: 3000,
            });
        } else {
            alert(`Server address saved locally: ${saved}`);
        }
    } catch (err) {
        alert(`Failed to save server URL: ${err.message}`);
    }
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

    // Server Configuration Info
    const cfgHead = document.createElement("h4");
    cfgHead.textContent = "Server Configuration (Saved to Disk)";
    cfgHead.style.margin = "12px 0 6px 0";
    cfgHead.style.color = "#61afef";
    container.appendChild(cfgHead);

    const cfgDesc = document.createElement("p");
    cfgDesc.style.fontSize = "12px";
    cfgDesc.textContent =
        "The Mac server address is saved locally in your ComfyUI configuration and never embedded in shared workflow files. " +
        "You can configure it via ComfyUI Settings or by clicking '⚙️ Configure Server Address' on this node.";
    container.appendChild(cfgDesc);

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

function installActionButtons(node) {
    // 1. Configure Server URL button
    node.addWidget("button", "⚙️ Configure Mac Server", null, async () => {
        await promptConfigureServerUrl();
    });

    // 2. Fetch Models button
    node.addWidget("button", "🔄 Fetch Models from Server", null, async () => {
        const modelNameWidget = node.widgets?.find(w => w.name === "model_name");
        if (!modelNameWidget) return;

        try {
            const proxyRes = await fetch("/macoffload/models");
            if (!proxyRes.ok) {
                const errBody = await proxyRes.json().catch(() => null);
                const msg = errBody?.error || `HTTP ${proxyRes.status}: ${proxyRes.statusText}`;
                throw new Error(msg);
            }

            const data = await proxyRes.json();
            const allModels = data.models || [];
            if (!allModels.length) {
                alert(
                    "Connected to Mac server, but no models were found in the server's models directory.\n\n" +
                    "Please verify that your model folders (e.g. Qwen3-VL-4b-Heretic) are placed directly inside the models directory on your Mac.\n" +
                    "Check the Mac server terminal output for the exact scanned directory."
                );
                return;
            }

            const modelTypes = data.model_types || {};
            const nodeClass = getNodeClass(node);
            const isFlux = nodeClass.includes("CLIPTextEncode") || nodeClass.includes("Flux");
            const isQwen = nodeClass.includes("QwenImage") || nodeClass.includes("Qwen");

            let filtered = allModels;
            if (isFlux) {
                const matched = allModels.filter(m => {
                    const type = modelTypes[m];
                    if (type === "flux_klein" || type === "flux3_klein") return true;
                    const lower = m.toLowerCase();
                    // Prioritize Qwen3-VL-4B as Flux.3 Klein text encoder even if server tagged it otherwise
                    if (lower.includes("qwen3") && lower.includes("4b")) return true;
                    if (type === "qwen_image") return false;
                    return lower.includes("flux");
                });
                if (matched.length > 0) filtered = matched;
            } else if (isQwen) {
                const matched = allModels.filter(m => {
                    const type = modelTypes[m];
                    const lower = m.toLowerCase();
                    // Exclude Qwen3-VL-4B (it belongs to Flux.3 Klein)
                    if (lower.includes("qwen3") && lower.includes("4b")) return false;
                    if (type === "qwen_image") return true;
                    if (type === "flux_klein" || type === "flux3_klein") return false;
                    return lower.includes("image") || (lower.includes("qwen") && !lower.includes("4b"));
                });
                if (matched.length > 0) filtered = matched;
            }

            console.log("[MacOffload] Fetched models from server:", {
                allModels,
                modelTypes,
                filtered,
                nodeClass,
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

            if (modelNameWidget.callback) {
                modelNameWidget.callback(modelNameWidget.value);
            }
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);

            if (app.extensionManager?.toast?.add) {
                app.extensionManager.toast.add({
                    severity: "success",
                    summary: "MacOffload",
                    detail: `Fetched ${targetList.length} models from server: ${targetList.join(", ")}`,
                    life: 4000,
                });
            }
        } catch (err) {
            alert(
                `Failed to fetch models:\n${err.message}\n\nTip: Click '⚙️ Configure Mac Server' to verify or update your Mac's LAN IP address.`
            );
        }
    });
}

app.registerExtension({
    name: "MacOffload.HelpBadge",
    async setup() {
        // Register setting in ComfyUI Settings panel if supported
        if (app.ui?.settings?.addSetting) {
            app.ui.settings.addSetting({
                id: "MacOffload.ServerUrl",
                name: "MacOffload: Remote macOS Server Address",
                type: "text",
                defaultValue: "http://127.0.0.1:8288",
                tooltip: "Base URL of the remote macOS te-server. Saved locally on disk, not inside workflows.",
                onChange: async (newVal) => {
                    if (newVal) {
                        try {
                            await saveServerUrl(newVal);
                        } catch (e) {
                            console.error("[MacOffload] Failed to persist server URL from settings:", e);
                        }
                    }
                },
            });
        }
    },
    nodeCreated(node) {
        if (!isMacOffloadNode(node)) return;
        installHelpBadge(node);
        installActionButtons(node);
    },
    getNodeMenuItems(node) {
        if (!isMacOffloadNode(node)) return [];
        return [
            {
                content: "⚙️ Configure Mac Server Address",
                callback: () => promptConfigureServerUrl(),
            },
            {
                content: "About this node ❓",
                callback: () => openHelpDialog(node),
            },
        ];
    },
});
