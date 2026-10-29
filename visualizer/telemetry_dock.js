// visualizer/telemetry_dock.js
// Scientific Canvas2D telemetry docking interface executing real-time cosmological metric plots.

export class ScientificTelemetryDock {
    constructor(canvasElement, wasmMemory, mmioOffset = 0) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.memory = wasmMemory;
        this.mmioOffset = mmioOffset;
        this.activeTab = 0; // 0: All/Overview, 1: P_m(k), 2: HMF, 3: f*sigma_8, 4: WHIM, 5: T-Web, 6: Debt Ledger
        this.tabs = [
            "Overview",
            "P_m(k)",
            "HMF dn/dlnM",
            "f\u03C3_8(z)",
            "WHIM Phase",
            "T-Web",
            "Debt Ledger"
        ];
        this.visible = true;

        // Interactive research tools state
        this.inspectorData = null; // 3D Raycast Object Inspector
        this.slabSlicer = { active: false, depth: 0.5, thickness: 0.05 }; // Key: S
        this.rsdMode = 0; // 0: Real-Space, 1: RSD (Redshift Space Distortion), 2: 2LPT (Key: R)
        this.rsdLabels = ["Real-Space", "RSD (v_pec)", "2LPT Displaced"];
        this.baoCaliper = { active: false, p1: null, p2: null, distance: 0 }; // Key: M
        this.splitLcdm = false; // Key: V
        this.renderMode = 0; // 0: Anisotropic Splats, 1: Volumetric WHIM, 2: Billboard Particles (Key: W)
        this.renderModeLabels = ["Anisotropic D5 Splats", "Volumetric WHIM", "Billboard Particles"];
        this.causalLifecycleActive = true; // Key: C

        this.mousePos = { x: 0, y: 0 };
        this.cachedData = {
            powerSpectrum: new Float32Array(256),
            hmf: new Float32Array(64),
            phaseSpace: new Uint32Array(4096),
            tweb: [0.45, 0.35, 0.15, 0.05], // Void, Sheet, Filament, Knot
            growthRate: []
        };

        this.setupEventListeners();
        this.resize();
    }

    setupEventListeners() {
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.canvas.width / rect.width;
            const scaleY = this.canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            // Tab bar click detection (y: 35 to 65)
            if (y >= 35 && y <= 65) {
                const tabWidth = (this.canvas.width - 40) / this.tabs.length;
                const clickedTab = Math.floor((x - 20) / tabWidth);
                if (clickedTab >= 0 && clickedTab < this.tabs.length) {
                    this.activeTab = clickedTab;
                }
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mousePos.x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
            this.mousePos.y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        });
    }

    updateMemory(wasmMemory, mmioOffset) {
        this.memory = wasmMemory;
        if (mmioOffset !== undefined) this.mmioOffset = mmioOffset;
    }

    resize(width = 640, height = 960) {
        this.canvas.width = width;
        this.canvas.height = height;
    }

    toggleVisibility() {
        this.visible = !this.visible;
        this.canvas.style.display = this.visible ? "block" : "none";
        return this.visible;
    }

    cycleRenderMode() {
        this.renderMode = (this.renderMode + 1) % this.renderModeLabels.length;
        return this.renderModeLabels[this.renderMode];
    }

    cycleRsdMode() {
        this.rsdMode = (this.rsdMode + 1) % this.rsdLabels.length;
        return this.rsdLabels[this.rsdMode];
    }

    toggleSlabSlicer() {
        this.slabSlicer.active = !this.slabSlicer.active;
        return this.slabSlicer.active;
    }

    adjustSlab(deltaDepth, deltaThickness = 0) {
        this.slabSlicer.depth = Math.max(0.0, Math.min(1.0, this.slabSlicer.depth + deltaDepth));
        if (deltaThickness !== 0) {
            this.slabSlicer.thickness = Math.max(0.01, Math.min(0.5, this.slabSlicer.thickness + deltaThickness));
        }
    }

    toggleBaoCaliper() {
        this.baoCaliper.active = !this.baoCaliper.active;
        if (!this.baoCaliper.active) {
            this.baoCaliper.p1 = null;
            this.baoCaliper.p2 = null;
            this.baoCaliper.distance = 0;
        }
        return this.baoCaliper.active;
    }

    setCaliperPoints(p1, p2, boxSizeMpc = 222.55) {
        this.baoCaliper.p1 = p1;
        this.baoCaliper.p2 = p2;
        if (p1 && p2) {
            const dx = (p1[0] - p2[0]) * boxSizeMpc;
            const dy = (p1[1] - p2[1]) * boxSizeMpc;
            const dz = (p1[2] - p2[2]) * boxSizeMpc;
            this.baoCaliper.distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    toggleSplitLcdm() {
        this.splitLcdm = !this.splitLcdm;
        return this.splitLcdm;
    }

    toggleCausalLifecycle() {
        this.causalLifecycleActive = !this.causalLifecycleActive;
        return this.causalLifecycleActive;
    }

    setInspectedObject(obj) {
        this.inspectorData = obj;
    }

    exportTelemetryData(currentMetrics = {}) {
        const exportBlob = {
            timestamp: new Date().toISOString(),
            metrics: currentMetrics,
            researchTools: {
                renderMode: this.renderModeLabels[this.renderMode],
                rsdMode: this.rsdLabels[this.rsdMode],
                slabSlicer: this.slabSlicer,
                baoCaliper: this.baoCaliper,
                splitLcdm: this.splitLcdm,
                causalLifecycle: this.causalLifecycleActive,
                inspectedObject: this.inspectorData
            },
            data: {
                powerSpectrum: Array.from(this.cachedData.powerSpectrum),
                hmf: Array.from(this.cachedData.hmf),
                twebFractions: this.cachedData.tweb,
            }
        };

        const jsonStr = JSON.stringify(exportBlob, null, 2);
        const blob = new Blob([jsonStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `shbt_cosmic_telemetry_z${(currentMetrics.z ?? 0).toFixed(2)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    renderMetrics(powerSpectrumBuffer, hmfBuffer, phaseSpaceBuffer, twebFractions, extraState = {}) {
        if (!this.visible) return;

        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.fillStyle = '#0a0d14';
        ctx.fillRect(0, 0, w, h);

        let magic = 0x54424853;
        let z = extraState.z ?? 0.0;
        let a = extraState.a ?? (1.0 / (1.0 + z));
        let debtGW = extraState.landauer_debt_gw ?? 0.0;
        let fLoad = extraState.f_load ?? 0.0;
        let hubble = extraState.hubble ?? 67.4;

        if (this.memory && this.memory.buffer && this.memory.buffer.byteLength >= this.mmioOffset + 64) {
            try {
                const headerView = new DataView(this.memory.buffer, this.mmioOffset, 128);
                magic = headerView.getUint32(0, true);
                if (magic === 0x54424853) {
                    z = headerView.getFloat32(28, true);
                    a = headerView.getFloat32(24, true);
                    debtGW = headerView.getFloat32(44, true);
                }
            } catch (err) {
                // fall back to extraState
            }
        }

        // Cache passed data if provided
        if (powerSpectrumBuffer && powerSpectrumBuffer.length) {
            this.cachedData.powerSpectrum = powerSpectrumBuffer;
        }
        if (hmfBuffer && hmfBuffer.length) {
            this.cachedData.hmf = hmfBuffer;
        }
        if (phaseSpaceBuffer && phaseSpaceBuffer.length) {
            this.cachedData.phaseSpace = phaseSpaceBuffer;
        }
        if (twebFractions) {
            this.cachedData.tweb = twebFractions;
        }

        // 1. Draw Title Header & Key Indicators
        this.drawPanelHeader(20, 25, `QUANTITATIVE COSMOLOGICAL TELEMETRY DOCK  |  z = ${z >= 1e4 ? z.toExponential(3) : z.toFixed(3)}  (a = ${a.toFixed(4)})`);

        // 2. Draw Tab Navigation Bar
        this.renderTabBar(20, 38, w - 40, 26);

        // 3. Render content depending on selected tab
        const contentY = 72;
        const contentH = h - contentY - 180;

        switch (this.activeTab) {
            case 0: // Overview (Multi-panel grid)
                this.renderPowerSpectrumPlot(20, contentY, 290, 220, this.cachedData.powerSpectrum);
                this.renderHMFPlot(330, contentY, 290, 220, this.cachedData.hmf);
                this.renderPhaseSpaceDensity(20, contentY + 235, 290, 220, this.cachedData.phaseSpace);
                this.renderTWebPlot(330, contentY + 235, 290, 220, this.cachedData.tweb);
                break;
            case 1: // Power Spectrum
                this.renderPowerSpectrumPlot(20, contentY, w - 40, contentH, this.cachedData.powerSpectrum, true);
                break;
            case 2: // HMF
                this.renderHMFPlot(20, contentY, w - 40, contentH, this.cachedData.hmf, true);
                break;
            case 3: // f*sigma_8 Growth Rate
                this.renderGrowthRatePlot(20, contentY, w - 40, contentH, z);
                break;
            case 4: // WHIM Phase Space
                this.renderPhaseSpaceDensity(20, contentY, w - 40, contentH, this.cachedData.phaseSpace, true);
                break;
            case 5: // T-Web Morphology
                this.renderTWebPlot(20, contentY, w - 40, contentH, this.cachedData.tweb, true);
                break;
            case 6: // Landauer Debt Ledger
                this.renderDebtLedger(20, contentY, w - 40, contentH, debtGW, z);
                break;
        }

        // 4. Research Tools Toolbar & Status Dock
        this.renderResearchToolsDock(20, h - 170, w - 40, 160, debtGW);
    }

    renderTabBar(x, y, w, h) {
        const ctx = this.ctx;
        const tabWidth = w / this.tabs.length;

        for (let i = 0; i < this.tabs.length; i++) {
            const tx = x + i * tabWidth;
            const isSelected = this.activeTab === i;

            ctx.fillStyle = isSelected ? '#1e3860' : '#101622';
            ctx.strokeStyle = isSelected ? '#00e5ff' : '#222d42';
            ctx.lineWidth = 1;

            ctx.beginPath();
            ctx.roundRect(tx + 2, y, tabWidth - 4, h, [4, 4, 0, 0]);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = isSelected ? '#ffffff' : '#8fa0c0';
            ctx.font = isSelected ? 'bold 10px monospace' : '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(this.tabs[i], tx + tabWidth / 2, y + 17);
        }
        ctx.textAlign = 'left';
    }

    renderPowerSpectrumPlot(x, y, w, h, data, detailed = false) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0c101a';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#1e2638';
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = '#8fa0c0';
        ctx.font = '11px monospace';
        ctx.fillText("3D Matter Power Spectrum P_m(k) [(Mpc/h)^3]", x + 10, y + 18);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(30, 45, 70, 0.5)';
        ctx.lineWidth = 0.5;
        for (let g = 1; g < 5; g++) {
            const gy = y + (g / 5) * h;
            ctx.beginPath();
            ctx.moveTo(x, gy);
            ctx.lineTo(x + w, gy);
            ctx.stroke();
        }

        // Plot data
        ctx.beginPath();
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 1.8;
        const nBins = data.length || 256;
        for (let i = 0; i < nBins; i++) {
            const px = x + (i / (nBins - 1)) * w;
            const val = data[i] > 0 ? data[i] : 1e-4;
            const logP = Math.log10(Math.max(val, 1e-4));
            const py = y + h - ((logP + 2.0) / 7.0) * h;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();

        if (detailed) {
            ctx.fillStyle = '#4fd8ff';
            ctx.font = '10px monospace';
            ctx.fillText("CIC Deconvolved \u00B7 Shot-Noise Subtracted \u00B7 k in [0.01, 10.0] h/Mpc", x + 10, y + h - 10);
            ctx.fillText("Log10 P_m(k) [h^-3 Mpc^3]", x + 10, y + 36);
        }
    }

    renderHMFPlot(x, y, w, h, hmfData, detailed = false) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0c101a';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#1e2638';
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = '#8fa0c0';
        ctx.font = '11px monospace';
        ctx.fillText("Cardy Halo Mass Function dn/dlnM [h^3 Mpc^-3]", x + 10, y + 18);

        ctx.beginPath();
        ctx.strokeStyle = '#ff9900';
        ctx.lineWidth = 1.8;
        const nBins = hmfData.length || 64;
        for (let i = 0; i < nBins; i++) {
            const px = x + (i / (nBins - 1)) * w;
            const val = hmfData[i] > 0 ? hmfData[i] : 1e-8;
            const logN = Math.log10(Math.max(val, 1e-8));
            const py = y + h - ((logN + 8.0) / 8.0) * h;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();

        if (detailed) {
            ctx.fillStyle = '#ffb42e';
            ctx.font = '10px monospace';
            ctx.fillText("Cardy c_eff = 1325/154 \u00B7 High-Mass Exponential Cutoff (NMS Basin)", x + 10, y + h - 10);
        }
    }

    renderPhaseSpaceDensity(x, y, w, h, histData, detailed = false) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0c101a';
        ctx.fillRect(x, y, w, h);

        const imgData = ctx.createImageData(64, 64);
        let maxCount = 1;
        for (let i = 0; i < 4096; i++) {
            if (histData[i] > maxCount) maxCount = histData[i];
        }

        for (let i = 0; i < 4096; i++) {
            const count = histData[i] || 0;
            const norm = count > 0 ? Math.log(count + 1) / Math.log(maxCount + 1) : 0;
            const val = Math.min(255, Math.floor(norm * 255));
            const base = i * 4;
            imgData.data[base + 0] = val;
            imgData.data[base + 1] = Math.floor(val * 0.4);
            imgData.data[base + 2] = 255 - val;
            imgData.data[base + 3] = count > 0 ? 255 : 30;
        }

        // Draw raster scaled to (w - 20, h - 35)
        createImageBitmap(imgData).then(bmp => {
            ctx.drawImage(bmp, x + 10, y + 25, w - 20, h - 35);
        }).catch(() => {});

        ctx.strokeStyle = '#1e2638';
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#8fa0c0';
        ctx.font = '11px monospace';
        ctx.fillText("WHIM Phase Space: log(1+\u03B4) vs log(T)", x + 10, y + 18);

        if (detailed) {
            ctx.fillStyle = '#33e0a1';
            ctx.font = '10px monospace';
            ctx.fillText("Warm-Hot Gas (10^5 - 10^7 K) \u00B7 Rankine-Hugoniot Accretion Shocks", x + 10, y + h - 6);
        }
    }

    renderTWebPlot(x, y, w, h, twebFractions = [0.45, 0.35, 0.15, 0.05], detailed = false) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0c101a';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#1e2638';
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = '#8fa0c0';
        ctx.font = '11px monospace';
        ctx.fillText("T-Web Morphology (\u03BB_th = 0.0198818)", x + 10, y + 18);

        const labels = ["Voids", "Sheets", "Filaments", "Knots"];
        const colors = ["#2b4a7f", "#33a0ff", "#00ffcc", "#ffaa00"];
        const barW = (w - 40) / 4;

        for (let i = 0; i < 4; i++) {
            const frac = twebFractions[i] ?? 0.25;
            const barH = (h - 60) * frac;
            const bx = x + 20 + i * barW;
            const by = y + h - 30 - barH;

            ctx.fillStyle = colors[i];
            ctx.fillRect(bx + 4, by, barW - 8, barH);

            ctx.fillStyle = '#ffffff';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${(frac * 100).toFixed(1)}%`, bx + barW / 2, by - 6);
            ctx.fillStyle = '#8fa0c0';
            ctx.fillText(labels[i], bx + barW / 2, y + h - 12);
        }
        ctx.textAlign = 'left';
    }

    renderGrowthRatePlot(x, y, w, h, currentZ) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0c101a';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#1e2638';
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = '#8fa0c0';
        ctx.font = '11px monospace';
        ctx.fillText("Linear Growth Rate f\u03C3_8(z) vs Redshift", x + 10, y + 18);

        // Theoretical curve f*sigma_8(z) ~ Omega_m(z)^0.55 * sigma_8(z)
        ctx.beginPath();
        ctx.strokeStyle = '#7ee0a3';
        ctx.lineWidth = 2.0;

        const zMax = 3.0;
        for (let i = 0; i < 100; i++) {
            const zVal = (i / 99) * zMax;
            const px = x + (i / 99) * w;
            const omZ = 0.30899 * Math.pow(1 + zVal, 3) / (0.30899 * Math.pow(1 + zVal, 3) + 0.691);
            const fZ = Math.pow(omZ, 0.55);
            const sig8Z = 0.8159 / (1 + zVal);
            const fsig8 = fZ * sig8Z;

            const py = y + h - (fsig8 / 0.6) * (h - 40) - 20;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Mark current redshift
        if (currentZ <= zMax && currentZ >= 0) {
            const curX = x + (currentZ / zMax) * w;
            ctx.strokeStyle = '#ff3366';
            ctx.lineWidth = 1.0;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(curX, y + 25);
            ctx.lineTo(curX, y + h - 10);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#ff3366';
            ctx.font = '10px monospace';
            ctx.fillText(`Current z = ${currentZ.toFixed(2)}`, curX + 5, y + 40);
        }
    }

    renderDebtLedger(x, y, w, h, debtGW, z) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0c101a';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#1e2638';
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = '#8fa0c0';
        ctx.font = '11px monospace';
        ctx.fillText("Landauer Information Debt Ledger & Thermodynamics", x + 10, y + 20);

        ctx.fillStyle = '#ffffff';
        ctx.font = '12px monospace';
        ctx.fillText(`\u2022 Cumulative Landauer Power Debt : ${debtGW.toFixed(2)} GW`, x + 20, y + 60);
        ctx.fillText(`\u2022 Seed Condensation Threshold    : 906.0 GW / M_\u2609`, x + 20, y + 90);
        ctx.fillText(`\u2022 Boundary Dissipation Rate      : k_B * T_horizon * ln(2) * dN/dt`, x + 20, y + 120);
        ctx.fillText(`\u2022 WZW Affine Branch Embedding   : su(2)_26 x su(3)_8 \u2282 so(10)_312`, x + 20, y + 150);
        ctx.fillText(`\u2022 Bit Capacity Ceiling N_sat     : 3.3119977 x 10^122 bits`, x + 20, y + 180);
        ctx.fillText(`\u2022 Central Charges                : c_matter = 1325/154, c_ambient = 43.875`, x + 20, y + 210);
    }

    renderResearchToolsDock(x, y, w, h, debtGW) {
        const ctx = this.ctx;
        ctx.fillStyle = '#0f1420';
        ctx.strokeStyle = '#26344d';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w, h);
        ctx.fillRect(x, y, w, h);

        ctx.fillStyle = '#00ffcc';
        ctx.font = 'bold 11px monospace';
        ctx.fillText("INTERACTIVE RESEARCH TOOLS & HOTKEYS", x + 10, y + 18);

        ctx.font = '10px monospace';
        ctx.fillStyle = '#cfe3ff';

        // Hotkey indicators
        const col1X = x + 10;
        const col2X = x + w / 2;

        ctx.fillText(`[P] Dock Toggle       : ${this.visible ? "ON" : "OFF"}`, col1X, y + 36);
        ctx.fillText(`[W] Render Mode       : ${this.renderModeLabels[this.renderMode]}`, col1X, y + 54);
        ctx.fillText(`[R] Redshift Space    : ${this.rsdLabels[this.rsdMode]}`, col1X, y + 72);
        ctx.fillText(`[S] 2D Slab Slicer    : ${this.slabSlicer.active ? "ACTIVE (z=" + this.slabSlicer.depth.toFixed(2) + ")" : "OFF"}`, col1X, y + 90);

        ctx.fillText(`[M] BAO Caliper       : ${this.baoCaliper.active ? (this.baoCaliper.distance > 0 ? this.baoCaliper.distance.toFixed(1) + " Mpc/h" : "Select 2 pts") : "OFF"}`, col2X, y + 36);
        ctx.fillText(`[V] Split SHBT/\u039BCDM   : ${this.splitLcdm ? "ACTIVE" : "OFF"}`, col2X, y + 54);
        ctx.fillText(`[C] Causal Lifecycle  : ${this.causalLifecycleActive ? "ACTIVE (5-Phase)" : "OFF"}`, col2X, y + 72);
        ctx.fillText(`[E] Export Metrics    : Ready (Press E to download)`, col2X, y + 90);

        // Active Object Inspector readout
        ctx.fillStyle = '#1e2638';
        ctx.fillRect(x + 10, y + 104, w - 20, 46);
        ctx.strokeStyle = '#334460';
        ctx.strokeRect(x + 10, y + 104, w - 20, 46);

        if (this.inspectorData) {
            ctx.fillStyle = '#00e5ff';
            ctx.fillText(`[Click-to-Inspect] Pos: [${this.inspectorData.pos.map(v => v.toFixed(2)).join(", ")}] | Type: ${this.inspectorData.type} | Mass: ${this.inspectorData.mass.toFixed(2)} M_\u2609`, x + 16, y + 122);
            ctx.fillStyle = '#a0c0e0';
            ctx.fillText(`Vel: [${this.inspectorData.vel.map(v => v.toFixed(1)).join(", ")}] km/s | T-Web: ${this.inspectorData.tweb || "Filament"} | P_debt: ${debtGW.toFixed(1)} GW`, x + 16, y + 138);
        } else {
            ctx.fillStyle = '#6f8db8';
            ctx.fillText("[Click-to-Inspect] Click anywhere on the 3D cosmic web to raycast & inspect local physical invariants", x + 16, y + 130);
        }
    }

    drawPanelHeader(x, y, title) {
        this.ctx.fillStyle = '#00ffcc';
        this.ctx.font = 'bold 12px monospace';
        this.ctx.fillText(title, x, y);
    }
}
