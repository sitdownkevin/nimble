import React, { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  Typography,
  Button,
  IconButton,
  Select,
  MenuItem,
  ListSubheader,
  Switch,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Tooltip,
  Dialog,
  DialogContent,
  Snackbar,
  Divider,
  Collapse,
  styled,
} from "@mui/material";
import {
  PlayArrowRounded,
  CompareArrowsRounded,
  RefreshRounded,
  DarkModeOutlined,
  LightModeOutlined,
  TuneRounded,
  FileUploadOutlined,
  DeleteOutlineRounded,
  ContentCopyRounded,
  KeyboardArrowDownRounded,
  KeyboardArrowRightRounded,
  OpenInFullRounded,
  AddRounded,
  CloseRounded,
  CheckRounded,
} from "@mui/icons-material";

/** Clean Google AI Studio model metadata (zero emojis, concise specs) */
const MODEL_META = {
  "nimble-4b-mm-sota": {
    name: "Nimble 4B · Multimodal SOTA",
    shortName: "Nimble 4B · Merger-LoRA",
    spec: "sandwich_v3 · Merger-LoRA · FP32",
    metric: "86.7% Test (TTA5) · 87.3% Val",
    group: "Nimble 4B",
  },
  "nimble-4b-mm-lora-all": {
    name: "Nimble 4B · Full Vision LoRA",
    shortName: "Nimble 4B · Full Vision",
    spec: "sandwich_v3 · LoRA-All · FP32",
    metric: "84.0% Single-view · 0.043 ECE",
    group: "Nimble 4B",
  },
  "nimble-4b-mm-frozen": {
    name: "Nimble 4B · Frozen Vision",
    shortName: "Nimble 4B · Frozen ViT",
    spec: "sandwich_v3 · Frozen ViT · FP32",
    metric: "86.0% Test · 0.408 NLL",
    group: "Nimble 4B",
  },
  "nimble-4b-text-sota": {
    name: "Nimble 4B · Text-only",
    shortName: "Nimble 4B · Text-only",
    spec: "sandwich_v3 · Ignores image",
    metric: "83.3% Test (text control)",
    group: "Nimble 4B",
  },
  "nimble-0.8b-mm-sota": {
    name: "Nimble 0.8B · Multimodal SOTA",
    shortName: "Nimble 0.8B · MM SOTA",
    spec: "sandwich_v3 · Frozen ViT · FP32",
    metric: "79.3% Single · 80.0% TTA5",
    group: "Nimble 0.8B",
  },
  "nimble-0.8b-rawad-mm": {
    name: "Nimble 0.8B · Raw-Ad Pixel",
    shortName: "Nimble 0.8B · Raw-Ad",
    spec: "Trained without OCR/Caption",
    metric: "76.0% Raw-Ad (+13.3pp vs text)",
    group: "Nimble 0.8B",
  },
  "nimble-0.8b-legacy-v1": {
    name: "Nimble 0.8B · Legacy v1",
    shortName: "Nimble 0.8B · Legacy v1",
    spec: "v1 double-JSON · BF16 head",
    metric: "76.0% Test · 1.542 NLL",
    group: "Legacy Baseline",
  },
  "nimble-0.8b-text-legacy-v1": {
    name: "Nimble 0.8B · Legacy Text v1",
    shortName: "Nimble 0.8B · Text v1",
    spec: "v1 JSON · Ignores image",
    metric: "77.3% Test · 1.234 NLL",
    group: "Legacy Baseline",
  },
  "kev-4b-mm": {
    name: "Kev 4B · Multimodal",
    shortName: "Kev 4B · PointerHead",
    spec: "FIM tokens · PointerHead",
    metric: "85.3% Test · 0.538 NLL",
    group: "Kev Architecture",
  },
  "kev-0.8b-mm": {
    name: "Kev 0.8B · Multimodal",
    shortName: "Kev 0.8B · PointerHead",
    spec: "FIM tokens · PointerHead",
    metric: "74.7% Test",
    group: "Kev Architecture",
  },
};

function stripEmoji(str = "") {
  return str.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "").trim();
}

function getModelDisplay(item) {
  if (MODEL_META[item.id]) return { id: item.id, ...MODEL_META[item.id] };
  return {
    id: item.id,
    name: stripEmoji(item.title || item.id),
    spec: item.group || "Model",
    metric: stripEmoji(item.note || ""),
    group: item.group || "Other",
  };
}

/** Authentic Material Design 3 enclosed pill switch */
const M3Switch = styled((props) => (
  <Switch focusVisibleClassName=".Mui-focusVisible" disableRipple {...props} />
))(({ theme }) => ({
  width: 38,
  height: 22,
  padding: 0,
  "& .MuiSwitch-switchBase": {
    padding: 0,
    margin: 3,
    transitionDuration: "180ms",
    "&.Mui-checked": {
      transform: "translateX(16px)",
      color: theme.palette.mode === "dark" ? "#062e6f" : "#ffffff",
      "& + .MuiSwitch-track": {
        backgroundColor: theme.palette.mode === "dark" ? "#a8c7fa" : "#0b57d0",
        opacity: 1,
        border: 0,
      },
    },
  },
  "& .MuiSwitch-thumb": {
    boxSizing: "border-box",
    width: 16,
    height: 16,
    boxShadow: "none",
  },
  "& .MuiSwitch-track": {
    borderRadius: 999,
    backgroundColor: theme.palette.mode === "dark" ? "#444746" : "#dfe3e8",
    border: `1px solid ${theme.palette.mode === "dark" ? "#8e918f" : "#c4c7c5"}`,
    opacity: 1,
    transition: theme.transitions.create(["background-color", "border-color"], { duration: 180 }),
  },
}));

function buildGoogleTheme(mode) {
  const dark = mode === "dark";
  return createTheme({
    palette: {
      mode,
      primary: {
        main: dark ? "#a8c7fa" : "#0b57d0",
        dark: dark ? "#8ab4f8" : "#0842a0",
        contrastText: dark ? "#062e6f" : "#ffffff",
      },
      background: {
        default: dark ? "#131314" : "#f8fafd",
        paper: dark ? "#1e1f20" : "#ffffff",
      },
      text: {
        primary: dark ? "#e3e3e3" : "#1f1f1f",
        secondary: dark ? "#9aa0a6" : "#5f6368",
      },
      divider: dark ? "#333537" : "#e3e8ef",
    },
    shape: { borderRadius: 12 },
    typography: {
      fontFamily:
        '"Google Sans", "Inter", -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
      button: { textTransform: "none", fontWeight: 500, letterSpacing: 0 },
    },
    components: {
      MuiPaper: {
        styleOverrides: { root: { backgroundImage: "none" } },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 999,
            boxShadow: "none",
            fontSize: "0.8125rem",
            paddingLeft: 16,
            paddingRight: 16,
            "&:hover": { boxShadow: "none" },
          },
        },
      },
      MuiTab: {
        styleOverrides: {
          root: {
            textTransform: "none",
            minHeight: 44,
            minWidth: "auto",
            padding: "0 14px",
            fontSize: "0.825rem",
            fontWeight: 500,
            color: dark ? "#9aa0a6" : "#5f6368",
            "&.Mui-selected": {
              color: dark ? "#a8c7fa" : "#0b57d0",
              fontWeight: 600,
            },
          },
        },
      },
      MuiTabs: {
        styleOverrides: {
          indicator: {
            height: 3,
            borderTopLeftRadius: 3,
            borderTopRightRadius: 3,
            backgroundColor: dark ? "#a8c7fa" : "#0b57d0",
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderColor: dark ? "#2a2b2d" : "#eef1f5",
            padding: "10px 14px",
          },
          head: {
            fontSize: "0.75rem",
            fontWeight: 500,
            color: dark ? "#9aa0a6" : "#5f6368",
            backgroundColor: "transparent",
            whiteSpace: "nowrap",
          },
          body: {
            fontSize: "0.8125rem",
          },
        },
      },
    },
  });
}

function withImagePos(state, uri, pos) {
  if (!uri || !state || typeof state !== "object" || Array.isArray(state)) return state;
  const entries = Object.entries(state).filter(([k]) => k !== "image");
  const idx = pos === "start" ? 0 : pos === "end" ? entries.length : Math.floor(entries.length / 2);
  entries.splice(idx, 0, ["image", uri]);
  return Object.fromEntries(entries);
}

/** Segmented pill control (Google M3 style) */
function SegmentedControl({ options, value, onChange, fullWidth = false }) {
  return (
    <Box
      sx={{
        display: fullWidth ? "grid" : "inline-flex",
        gridTemplateColumns: fullWidth ? `repeat(${options.length}, 1fr)` : undefined,
        p: "3px",
        borderRadius: 999,
        bgcolor: (t) => (t.palette.mode === "dark" ? "#282a2c" : "#eef2f7"),
        gap: "2px",
      }}
    >
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <Box
            key={opt.value}
            component="button"
            type="button"
            onClick={() => onChange(opt.value)}
            sx={{
              border: 0,
              cursor: "pointer",
              px: 1.4,
              py: 0.55,
              borderRadius: 999,
              fontFamily: "inherit",
              fontSize: "0.75rem",
              fontWeight: active ? 600 : 500,
              whiteSpace: "nowrap",
              transition: "all 140ms ease",
              color: (t) =>
                active
                  ? t.palette.text.primary
                  : t.palette.text.secondary,
              bgcolor: (t) =>
                active
                  ? t.palette.mode === "dark"
                    ? "#3c4043"
                    : "#ffffff"
                  : "transparent",
              boxShadow: active ? "0 1px 2px rgba(0,0,0,0.08)" : "none",
              "&:hover": {
                color: "text.primary",
              },
            }}
          >
            {opt.label}
          </Box>
        );
      })}
    </Box>
  );
}

/** Quiet, minimal code editor section */
function MinimalJsonEditor({ label, metaText, value, onChange, error, rightActions, sx }) {
  const gutterRef = useRef(null);
  const lineCount = useMemo(() => (value.match(/\n/g) || []).length + 1, [value]);
  const lineNumbers = useMemo(
    () => Array.from({ length: lineCount }, (_, i) => i + 1).join("\n"),
    [lineCount]
  );

  const handleKeyDown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const el = e.target;
      const start = el.selectionStart;
      const end = el.selectionEnd;
      const next = value.slice(0, start) + "  " + value.slice(end);
      onChange(next);
      requestAnimationFrame(() => {
        el.selectionStart = el.selectionEnd = start + 2;
      });
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, ...sx }}>
      <Box
        sx={{
          height: 34,
          px: 2,
          display: "flex",
          alignItems: "center",
          gap: 1,
          borderTop: 1,
          borderBottom: 1,
          borderColor: "divider",
          bgcolor: (t) => (t.palette.mode === "dark" ? "#191a1c" : "#fafbfd"),
          flexShrink: 0,
        }}
      >
        <Typography sx={{ fontSize: "0.76rem", fontWeight: 600, color: "text.primary" }}>
          {label}
        </Typography>
        {metaText && !error && (
          <Typography sx={{ fontSize: "0.72rem", color: "text.secondary" }}>
            {metaText}
          </Typography>
        )}
        {error && (
          <Typography
            sx={{
              fontSize: "0.72rem",
              fontWeight: 500,
              color: (t) => (t.palette.mode === "dark" ? "#f28b82" : "#b3261e"),
            }}
          >
            JSON 语法错误
          </Typography>
        )}
        <Box sx={{ flex: 1 }} />
        {rightActions}
      </Box>

      <Box sx={{ display: "flex", flex: 1, minHeight: 0 }}>
        <Box
          ref={gutterRef}
          component="pre"
          sx={{
            m: 0,
            py: 1.25,
            px: 1,
            width: 36,
            textAlign: "right",
            userSelect: "none",
            overflow: "hidden",
            fontFamily: '"Google Sans Mono", "SF Mono", "JetBrains Mono", Menlo, monospace',
            fontSize: "12px",
            lineHeight: "1.6",
            color: (t) => (t.palette.mode === "dark" ? "#5f6368" : "#9aa0a6"),
            flexShrink: 0,
          }}
        >
          {lineNumbers}
        </Box>
        <Box
          component="textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onScroll={(e) => {
            if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop;
          }}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          sx={{
            flex: 1,
            width: "100%",
            border: 0,
            outline: "none",
            resize: "none",
            py: 1.25,
            pr: 2,
            pl: 0.5,
            m: 0,
            bgcolor: "transparent",
            color: "text.primary",
            fontFamily: '"Google Sans Mono", "SF Mono", "JetBrains Mono", Menlo, monospace',
            fontSize: "12.5px",
            lineHeight: "1.6",
            overflow: "auto",
          }}
        />
      </Box>
    </Box>
  );
}

/** Probability distribution view for a single question */
function ProbabilityDistribution({ qid, ans, gold }) {
  if (!ans) return null;

  if (ans.type === "choice") {
    const sorted = Object.entries(ans.probabilities || {}).sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, 8);
    return (
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", mb: 1.25 }}>
          <Typography sx={{ fontSize: "0.78rem", fontWeight: 600, color: "text.secondary" }}>
            候选概率分布 · {qid}
          </Typography>
          <Typography sx={{ fontSize: "0.74rem", color: "text.secondary" }}>
            显示前 {top.length} / {sorted.length} 项
            {ans.confidence != null ? ` · 归一化置信度 ${(ans.confidence * 100).toFixed(1)}%` : ""}
          </Typography>
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.9 }}>
          {top.map(([label, prob], idx) => {
            const isGold = Boolean(gold && label === gold);
            const isWrongTop = Boolean(gold && idx === 0 && label !== gold);
            return (
              <Box
                key={label}
                sx={{
                  display: "grid",
                  gridTemplateColumns: "20px minmax(150px, 210px) 1fr 56px",
                  alignItems: "center",
                  gap: 1.25,
                  py: 0.35,
                }}
              >
                <Typography
                  sx={{
                    fontSize: "0.73rem",
                    fontFamily: "monospace",
                    color: "text.secondary",
                    textAlign: "right",
                  }}
                >
                  {idx + 1}
                </Typography>

                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, minWidth: 0 }}>
                  <Typography
                    noWrap
                    title={label}
                    sx={{
                      fontSize: "0.8125rem",
                      fontWeight: idx === 0 || isGold ? 600 : 400,
                      color: "text.primary",
                    }}
                  >
                    {label}
                  </Typography>
                  {isGold && (
                    <Box
                      component="span"
                      sx={{
                        px: 0.7,
                        py: "1px",
                        borderRadius: "4px",
                        fontSize: "0.68rem",
                        fontWeight: 600,
                        flexShrink: 0,
                        bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(129,201,149,0.16)" : "#e6f4ea"),
                        color: (t) => (t.palette.mode === "dark" ? "#81c995" : "#146c2e"),
                      }}
                    >
                      参考
                    </Box>
                  )}
                </Box>

                <Box
                  sx={{
                    height: 6,
                    borderRadius: 999,
                    overflow: "hidden",
                    bgcolor: (t) => (t.palette.mode === "dark" ? "#2c2d30" : "#eef2f6"),
                  }}
                >
                  <Box
                    sx={{
                      height: "100%",
                      width: `${Math.min(100, Math.max(0.8, prob * 100))}%`,
                      borderRadius: 999,
                      transition: "width 240ms cubic-bezier(0.2, 0, 0, 1)",
                      bgcolor: (t) => {
                        if (isGold) return t.palette.mode === "dark" ? "#81c995" : "#1e8e3e";
                        if (isWrongTop) return t.palette.mode === "dark" ? "#f28b82" : "#d93025";
                        if (idx === 0) return t.palette.primary.main;
                        return t.palette.mode === "dark" ? "#5f6368" : "#bdc1c6";
                      },
                    }}
                  />
                </Box>

                <Typography
                  sx={{
                    fontSize: "0.8rem",
                    textAlign: "right",
                    fontVariantNumeric: "tabular-nums",
                    fontFamily: '"Google Sans Mono", "SF Mono", Menlo, monospace',
                    fontWeight: idx === 0 || isGold ? 600 : 400,
                    color: idx === 0 || isGold ? "text.primary" : "text.secondary",
                  }}
                >
                  {(prob * 100).toFixed(1)}%
                </Typography>
              </Box>
            );
          })}
        </Box>
      </Box>
    );
  }

  if (ans.type === "noul") {
    const pTrue = ans.probabilities?.true ?? ans.noul ?? 0;
    const pFalse = ans.probabilities?.false ?? 1 - pTrue;
    return (
      <Box sx={{ mb: 3 }}>
        <Typography sx={{ fontSize: "0.78rem", fontWeight: 600, color: "text.secondary", mb: 1.25 }}>
          布尔判定 · {qid}
        </Typography>
        {[
          ["true", pTrue],
          ["false", pFalse],
        ].map(([k, p]) => (
          <Box
            key={k}
            sx={{ display: "grid", gridTemplateColumns: "90px 1fr 56px", alignItems: "center", gap: 1.5, py: 0.4 }}
          >
            <Typography sx={{ fontSize: "0.8125rem", fontFamily: "monospace", fontWeight: 500 }}>{k}</Typography>
            <Box sx={{ height: 6, borderRadius: 999, bgcolor: (t) => (t.palette.mode === "dark" ? "#2c2d30" : "#eef2f6") }}>
              <Box
                sx={{
                  height: "100%",
                  width: `${Math.max(0.8, p * 100)}%`,
                  borderRadius: 999,
                  bgcolor: "primary.main",
                }}
              />
            </Box>
            <Typography sx={{ fontSize: "0.8rem", textAlign: "right", fontFamily: "monospace" }}>
              {(p * 100).toFixed(1)}%
            </Typography>
          </Box>
        ))}
      </Box>
    );
  }

  const probs = Object.entries(ans.probabilities || {});
  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1.25 }}>
        <Typography sx={{ fontSize: "0.78rem", fontWeight: 600, color: "text.secondary" }}>
          等级评分 · {qid}
        </Typography>
        <Typography sx={{ fontSize: "0.76rem", fontWeight: 600 }}>期望分值 {ans.score}</Typography>
      </Box>
      {probs.map(([k, p]) => (
        <Box
          key={k}
          sx={{ display: "grid", gridTemplateColumns: "210px 1fr 56px", alignItems: "center", gap: 1.5, py: 0.4 }}
        >
          <Typography noWrap sx={{ fontSize: "0.8rem" }}>
            {ans.legend?.[k] || k}
          </Typography>
          <Box sx={{ height: 6, borderRadius: 999, bgcolor: (t) => (t.palette.mode === "dark" ? "#2c2d30" : "#eef2f6") }}>
            <Box sx={{ height: "100%", width: `${Math.max(0.8, p * 100)}%`, borderRadius: 999, bgcolor: "primary.main" }} />
          </Box>
          <Typography sx={{ fontSize: "0.8rem", textAlign: "right", fontFamily: "monospace" }}>
            {(p * 100).toFixed(1)}%
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

export default function App() {
  const [themeMode, setThemeMode] = useState("light");
  const theme = useMemo(() => buildGoogleTheme(themeMode), [themeMode]);

  const [catalog, setCatalog] = useState([]);
  const [sampleCounts, setSampleCounts] = useState({ all: 150, disagree: 25, poster_only: 33 });
  const [sampleMode, setSampleMode] = useState("disagree");
  const [selectedModel, setSelectedModel] = useState("nimble-4b-mm-sota");
  const [settingsOpen, setSettingsOpen] = useState(true);

  const [currentSample, setCurrentSample] = useState(null);
  const [savedFullState, setSavedFullState] = useState(null);
  const [modalityMode, setModalityMode] = useState("full");

  const [imageUri, setImageUri] = useState(null);
  const [imageName, setImageName] = useState("");
  const [imgPos, setImgPos] = useState("start");
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const [stateText, setStateText] = useState("{}");
  const [questionsText, setQuestionsText] = useState("{}");
  const [stateErr, setStateErr] = useState("");
  const [questionsErr, setQuestionsErr] = useState("");

  const [calibrated, setCalibrated] = useState(true);
  const [tta5, setTta5] = useState(false);

  const [activeTab, setActiveTab] = useState(0);
  const [loadingSingle, setLoadingSingle] = useState(false);
  const [loadingCompare, setLoadingCompare] = useState(false);
  const [loadingPrompt, setLoadingPrompt] = useState(false);

  const [singleResult, setSingleResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [promptResult, setPromptResult] = useState(null);
  const [rawJsonOpen, setRawJsonOpen] = useState(false);
  const [tokenTableOpen, setTokenTableOpen] = useState(false);
  const [showAllTokens, setShowAllTokens] = useState(false);
  const [toast, setToast] = useState("");

  const fileInputRef = useRef(null);

  const displayCatalog = useMemo(() => catalog.map(getModelDisplay), [catalog]);
  const currentModelMeta = useMemo(
    () => displayCatalog.find((m) => m.id === selectedModel) || MODEL_META[selectedModel] || { name: selectedModel },
    [displayCatalog, selectedModel]
  );

  const groupedCatalog = useMemo(() => {
    const groups = new Map();
    for (const m of displayCatalog) {
      if (!groups.has(m.group)) groups.set(m.group, []);
      groups.get(m.group).push(m);
    }
    return Array.from(groups.entries());
  }, [displayCatalog]);

  const validateJson = useCallback((txt, setter) => {
    try {
      JSON.parse(txt);
      setter("");
      return true;
    } catch (e) {
      setter(e.message);
      return false;
    }
  }, []);

  const updateStateEditor = useCallback(
    (objOrStr, modeTag = "custom") => {
      const str = typeof objOrStr === "string" ? objOrStr : JSON.stringify(objOrStr, null, 2);
      setStateText(str);
      validateJson(str, setStateErr);
      setModalityMode(modeTag);
    },
    [validateJson]
  );

  const updateQuestionsEditor = useCallback(
    (objOrStr) => {
      const str = typeof objOrStr === "string" ? objOrStr : JSON.stringify(objOrStr, null, 2);
      setQuestionsText(str);
      validateJson(str, setQuestionsErr);
    },
    [validateJson]
  );

  const executeSingleDirect = useCallback(async (stObj, qObj, imgData, modelId, cal, useTta5, pos) => {
    const finalState = imgData ? withImagePos(stObj, imgData, pos) : stObj;
    const body = JSON.stringify({
      state: finalState,
      questions: qObj,
      model: modelId,
      calibrated: cal,
      tta_views: useTta5 ? 5 : 1,
    });
    const headers = { "Content-Type": "application/json" };
    setLoadingSingle(true);
    try {
      const [ra, rp] = await Promise.all([
        fetch("/v1/systemone", { method: "POST", headers, body }),
        fetch("/api/prompt", { method: "POST", headers, body }),
      ]);
      const ja = await ra.json().catch(() => ({ detail: ra.statusText }));
      const jp = await rp.json().catch(() => ({ detail: rp.statusText }));
      setSingleResult({ status: ra.status, body: ja });
      if (rp.ok) setPromptResult(jp);
    } catch (e) {
      setToast("推理请求失败: " + String(e));
    } finally {
      setLoadingSingle(false);
    }
  }, []);

  const fetchRandomSample = useCallback(
    async (modeOverride, modelOverride) => {
      const m = modeOverride ?? sampleMode;
      const targetModel = modelOverride ?? selectedModel;
      try {
        const res = await fetch(`/api/sample/random?mode=${encodeURIComponent(m)}`);
        const s = await res.json();
        setCurrentSample(s);
        setSavedFullState(s.state);
        updateStateEditor(s.state, "full");
        updateQuestionsEditor(s.questions);
        setImageUri(s.image);
        setImageName(`${s.id}.jpg`);
        setCompareResult(null);
        setShowAllTokens(false);
        // Immediately run ~180ms inference so Output pane always shows live calibrated probabilities
        await executeSingleDirect(s.state, s.questions, s.image, targetModel, calibrated, tta5, imgPos);
      } catch (e) {
        setToast("样本加载失败: " + String(e));
      }
    },
    [sampleMode, selectedModel, calibrated, tta5, imgPos, updateStateEditor, updateQuestionsEditor, executeSingleDirect]
  );

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/catalog");
        const cat = await res.json();
        const firstModel = cat.models?.[0]?.id || "nimble-4b-mm-sota";
        if (cat.models?.length) {
          setCatalog(cat.models);
          setSelectedModel(firstModel);
        }
        if (cat.sample_counts) setSampleCounts(cat.sample_counts);
        await fetchRandomSample("disagree", firstModel);
      } catch (e) {
        setToast("初始化失败: " + String(e));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const buildPayload = useCallback(() => {
    let stObj, qObj;
    try {
      stObj = JSON.parse(stateText);
    } catch (e) {
      setToast("State JSON 语法错误: " + e.message);
      return null;
    }
    try {
      qObj = JSON.parse(questionsText);
    } catch (e) {
      setToast("Questions JSON 语法错误: " + e.message);
      return null;
    }
    return {
      state: imageUri ? withImagePos(stObj, imageUri, imgPos) : stObj,
      questions: qObj,
      calibrated,
      tta_views: tta5 ? 5 : 1,
    };
  }, [stateText, questionsText, imageUri, imgPos, calibrated, tta5]);

  const runSingleModel = useCallback(
    async (modelOverride) => {
      const payload = buildPayload();
      if (!payload) return;
      const targetModel = typeof modelOverride === "string" ? modelOverride : selectedModel;
      const body = JSON.stringify({ ...payload, model: targetModel });
      const headers = { "Content-Type": "application/json" };
      setLoadingSingle(true);
      try {
        const [ra, rp] = await Promise.all([
          fetch("/v1/systemone", { method: "POST", headers, body }),
          fetch("/api/prompt", { method: "POST", headers, body }),
        ]);
        const ja = await ra.json().catch(() => ({ detail: ra.statusText }));
        const jp = await rp.json().catch(() => ({ detail: rp.statusText }));
        setSingleResult({ status: ra.status, body: ja });
        if (rp.ok) setPromptResult(jp);
        if (activeTab === 1 && !modelOverride) setActiveTab(0);
      } catch (e) {
        setToast("推理失败: " + String(e));
      } finally {
        setLoadingSingle(false);
      }
    },
    [buildPayload, selectedModel, activeTab]
  );

  const fetchPromptOnly = useCallback(
    async (targetModel) => {
      const payload = buildPayload();
      if (!payload) return;
      setSelectedModel(targetModel);
      setActiveTab(2);
      setShowAllTokens(false);
      setLoadingPrompt(true);
      try {
        const rp = await fetch("/api/prompt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...payload, model: targetModel }),
        });
        const jp = await rp.json();
        if (rp.ok) setPromptResult(jp);
      } catch (e) {
        setToast("Token 视图加载失败: " + String(e));
      } finally {
        setLoadingPrompt(false);
      }
    },
    [buildPayload]
  );

  const runCompareAll = useCallback(async () => {
    const payload = buildPayload();
    if (!payload) return;
    setActiveTab(1);
    setLoadingCompare(true);
    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setCompareResult(await res.json());
    } catch (e) {
      setToast("对比评测失败: " + String(e));
    } finally {
      setLoadingCompare(false);
    }
  }, [buildPayload]);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        runSingleModel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runSingleModel]);

  const handleModalityChange = (nextMode) => {
    let baseState = savedFullState;
    if (!baseState) {
      try {
        baseState = JSON.parse(stateText);
        setSavedFullState(baseState);
      } catch {
        return;
      }
    }
    if (nextMode === "full") {
      updateStateEditor(baseState, "full");
    } else if (nextMode === "raw_ad") {
      const stripped = {};
      for (const k of ["title", "body"]) {
        if (baseState[k]) stripped[k] = baseState[k];
      }
      updateStateEditor(stripped, "raw_ad");
    } else if (nextMode === "pure_image") {
      updateStateEditor({}, "pure_image");
    }
  };

  const handleUploadImage = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      setCurrentSample(null);
      setImageUri(reader.result);
      setImageName(f.name);
      setToast(`已载入图片 ${f.name}`);
    };
    reader.readAsDataURL(f);
    e.target.value = "";
  };

  const addQuestionPreset = (kind) => {
    try {
      const q = JSON.parse(questionsText);
      if (kind === "noul") {
        let i = 1;
        while (q[`risk_check_${i}`]) i++;
        q[`risk_check_${i}`] = {
          type: "noul",
          instructions: "该广告画面或文案中是否包含夸大绝对化用语或功效承诺风险？",
          criteria: { true: "包含风险暗示或承诺", false: "客观合规展示" },
        };
      } else {
        let i = 1;
        while (q[`clarity_${i}`]) i++;
        q[`clarity_${i}`] = {
          type: "score",
          instructions: "评估广告素材对目标行业意图表达的清晰度等级",
          criteria: ["0分：意图模糊", "1分：弱相关线索", "2分：主体明确", "3分：行业高度清晰"],
        };
      }
      updateQuestionsEditor(q);
    } catch {
      setToast("请先修正 Questions JSON 语法");
    }
  };

  const formatJsonEditors = () => {
    try {
      setStateText(JSON.stringify(JSON.parse(stateText), null, 2));
      setStateErr("");
    } catch {}
    try {
      setQuestionsText(JSON.stringify(JSON.parse(questionsText), null, 2));
      setQuestionsErr("");
    } catch {}
  };

  const stateFieldCount = useMemo(() => {
    try {
      const o = JSON.parse(stateText);
      return o && typeof o === "object" && !Array.isArray(o) ? `${Object.keys(o).length} 个字段` : "";
    } catch {
      return "";
    }
  }, [stateText]);

  const questionCountText = useMemo(() => {
    try {
      const o = JSON.parse(questionsText);
      return o && typeof o === "object" ? `${Object.keys(o).length} 个问题` : "";
    } catch {
      return "";
    }
  }, [questionsText]);

  const compareKpis = useMemo(() => {
    if (!compareResult?.results?.length) return null;
    const gold = currentSample?.gold;
    let qid = "industry";
    try {
      qid = Object.keys(JSON.parse(questionsText))[0] || "industry";
    } catch {}
    const okItems = compareResult.results.filter((x) => x.ok);
    const correctCount = gold
      ? okItems.filter((x) => x.result?.answers?.[qid]?.choice === gold).length
      : null;

    const findModel = (id) => okItems.find((x) => x.model === id)?.result;
    const sota4b = findModel("nimble-4b-mm-sota");
    const text4b = findModel("nimble-4b-text-sota");
    const leg08 = findModel("nimble-0.8b-legacy-v1");
    const sota08 = findModel("nimble-0.8b-mm-sota");

    const pGoldSota4b = gold ? sota4b?.answers?.[qid]?.probabilities?.[gold] : null;
    const pGoldText4b = gold ? text4b?.answers?.[qid]?.probabilities?.[gold] : null;
    const tokV3 = (sota08 || sota4b)?.usage?.input_tokens;
    const tokV1 = leg08?.usage?.input_tokens;

    return { qid, gold, total: okItems.length, correctCount, pGoldSota4b, pGoldText4b, tokV3, tokV1 };
  }, [compareResult, currentSample, questionsText]);

  const isBusy = loadingSingle || loadingCompare || loadingPrompt;
  const gold = currentSample?.gold;
  const singleIndustryAns = singleResult?.body?.answers?.industry;
  const singleIsCorrect = gold && singleIndustryAns?.choice ? singleIndustryAns.choice === gold : null;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          bgcolor: "background.default",
          overflow: "hidden",
        }}
      >
        {/* Google AI Studio Top Bar */}
        <Box
          component="header"
          sx={{
            height: 54,
            px: 2,
            display: "flex",
            alignItems: "center",
            gap: 1.25,
            flexShrink: 0,
          }}
        >
          {/* Minimal Geometric Brand Mark */}
          <Box
            sx={{
              width: 24,
              height: 24,
              borderRadius: "7px",
              bgcolor: "primary.main",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Box
              sx={{
                width: 9,
                height: 9,
                borderRadius: "2px",
                bgcolor: (t) => (t.palette.mode === "dark" ? "#062e6f" : "#ffffff"),
                transform: "rotate(45deg)",
              }}
            />
          </Box>

          <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.75 }}>
            <Typography sx={{ fontSize: "1rem", fontWeight: 600, letterSpacing: "-0.01em", color: "text.primary" }}>
              Nimble
            </Typography>
            <Typography sx={{ fontSize: "0.95rem", fontWeight: 400, color: "text.secondary" }}>
              Studio
            </Typography>
          </Box>

          {currentSample && (
            <>
              <Divider orientation="vertical" flexItem sx={{ my: 1.75, mx: 0.5 }} />
              <Typography
                sx={{
                  fontSize: "0.78rem",
                  fontFamily: '"Google Sans Mono", "SF Mono", Menlo, monospace',
                  color: "text.secondary",
                  display: { xs: "none", md: "block" },
                }}
              >
                {currentSample.id}
              </Typography>
            </>
          )}

          <Box sx={{ flex: 1 }} />

          {/* Sample Pool Pill Selector */}
          <Select
            size="small"
            value={sampleMode}
            onChange={(e) => {
              setSampleMode(e.target.value);
              fetchRandomSample(e.target.value);
            }}
            sx={{
              height: 34,
              borderRadius: 999,
              bgcolor: "background.paper",
              fontSize: "0.8rem",
              "& .MuiOutlinedInput-notchedOutline": { borderColor: "divider" },
              "& .MuiSelect-select": { pl: 1.75, pr: "32px !important", py: 0.6 },
            }}
          >
            <MenuItem value="disagree" sx={{ fontSize: "0.8125rem" }}>
              分歧样本 · {sampleCounts.disagree}（SOTA 修正）
            </MenuItem>
            <MenuItem value="poster_only" sx={{ fontSize: "0.8125rem" }}>
              纯海报样本 · {sampleCounts.poster_only}（无文案）
            </MenuItem>
            <MenuItem value="all" sx={{ fontSize: "0.8125rem" }}>
              全部测试集 · {sampleCounts.all}
            </MenuItem>
          </Select>

          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshRounded sx={{ fontSize: 17 }} />}
            onClick={() => fetchRandomSample()}
            sx={{
              height: 34,
              bgcolor: "background.paper",
              borderColor: "divider",
              color: "text.primary",
              "&:hover": { borderColor: "text.secondary", bgcolor: "background.paper" },
            }}
          >
            随机抽样
          </Button>

          {/* M3 Tonal Button for All-Model Benchmark */}
          <Button
            size="small"
            disabled={isBusy}
            startIcon={<CompareArrowsRounded sx={{ fontSize: 17 }} />}
            onClick={runCompareAll}
            sx={{
              height: 34,
              bgcolor: (t) => (t.palette.mode === "dark" ? "#0842a0" : "#d3e3fd"),
              color: (t) => (t.palette.mode === "dark" ? "#d3e3fd" : "#041e49"),
              fontWeight: 600,
              "&:hover": {
                bgcolor: (t) => (t.palette.mode === "dark" ? "#0b57d0" : "#c2d7fc"),
              },
            }}
          >
            全模型对比
          </Button>

          {/* M3 Filled Primary Pill Button */}
          <Button
            variant="contained"
            color="primary"
            size="small"
            disabled={isBusy}
            startIcon={<PlayArrowRounded sx={{ fontSize: 18 }} />}
            onClick={() => runSingleModel()}
            sx={{ height: 34, fontWeight: 600 }}
          >
            运行 ⌘↵
          </Button>

          <Divider orientation="vertical" flexItem sx={{ my: 1.5, mx: 0.25 }} />

          <Tooltip title={settingsOpen ? "收起配置栏" : "展开配置栏"}>
            <IconButton
              size="small"
              onClick={() => setSettingsOpen((v) => !v)}
              sx={{
                width: 34,
                height: 34,
                color: settingsOpen ? "primary.main" : "text.secondary",
                bgcolor: settingsOpen
                  ? (t) => (t.palette.mode === "dark" ? "rgba(168,199,250,0.12)" : "#e8f0fe")
                  : "transparent",
              }}
            >
              <TuneRounded sx={{ fontSize: 19 }} />
            </IconButton>
          </Tooltip>

          <Tooltip title={themeMode === "light" ? "深色外观" : "浅色外观"}>
            <IconButton
              size="small"
              onClick={() => setThemeMode((m) => (m === "light" ? "dark" : "light"))}
              sx={{ width: 34, height: 34, color: "text.secondary" }}
            >
              {themeMode === "light" ? <DarkModeOutlined sx={{ fontSize: 19 }} /> : <LightModeOutlined sx={{ fontSize: 19 }} />}
            </IconButton>
          </Tooltip>
        </Box>

        {/* Studio Main Body: Floating Split Canvas + Right Run Settings Rail */}
        <Box
          sx={{
            flex: 1,
            minHeight: 0,
            px: 1.5,
            pb: 1.5,
            display: "flex",
            gap: 1.5,
            overflow: "hidden",
          }}
        >
          {/* Center Floating Rounded Surface */}
          <Box
            sx={{
              flex: 1,
              minWidth: 0,
              bgcolor: "background.paper",
              borderRadius: "16px",
              border: 1,
              borderColor: "divider",
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "43% 57%" },
              overflow: "hidden",
            }}
          >
            {/* Left Workspace Column: Input */}
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                minHeight: 0,
                borderRight: { md: 1 },
                borderColor: "divider",
              }}
            >
              {/* Input Pane Header */}
              <Box
                sx={{
                  height: 44,
                  px: 2,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  borderBottom: 1,
                  borderColor: "divider",
                  flexShrink: 0,
                }}
              >
                <Typography sx={{ fontSize: "0.825rem", fontWeight: 600 }}>输入上下文</Typography>
                <SegmentedControl
                  value={modalityMode}
                  onChange={handleModalityChange}
                  options={[
                    { value: "full", label: "完整输入" },
                    { value: "raw_ad", label: "去 OCR/描述" },
                    { value: "pure_image", label: "纯图像" },
                  ]}
                />
              </Box>

              {/* Ad Creative Stage (Uncropped `contain` preview!) */}
              <Box sx={{ p: 1.75, flexShrink: 0 }}>
                <Box
                  sx={{
                    display: "flex",
                    gap: 2,
                    p: 1.25,
                    borderRadius: "12px",
                    bgcolor: (t) => (t.palette.mode === "dark" ? "#161719" : "#f8fafd"),
                    border: 1,
                    borderColor: "divider",
                    alignItems: "center",
                  }}
                >
                  <Box
                    onClick={() => imageUri && setLightboxOpen(true)}
                    sx={{
                      width: 176,
                      height: 120,
                      borderRadius: "8px",
                      bgcolor: (t) => (t.palette.mode === "dark" ? "#0e0f10" : "#eef2f6"),
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      overflow: "hidden",
                      position: "relative",
                      cursor: imageUri ? "zoom-in" : "default",
                      flexShrink: 0,
                      "&:hover .img-expand": { opacity: 1 },
                    }}
                  >
                    {imageUri ? (
                      <>
                        <Box
                          component="img"
                          src={imageUri}
                          alt="广告素材"
                          sx={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                        />
                        <Box
                          className="img-expand"
                          sx={{
                            position: "absolute",
                            right: 6,
                            bottom: 6,
                            width: 24,
                            height: 24,
                            borderRadius: "6px",
                            bgcolor: "rgba(0,0,0,0.58)",
                            color: "#fff",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            opacity: 0,
                            transition: "opacity 150ms",
                          }}
                        >
                          <OpenInFullRounded sx={{ fontSize: 13 }} />
                        </Box>
                      </>
                    ) : (
                      <Typography sx={{ fontSize: "0.75rem", color: "text.secondary" }}>未附带图像</Typography>
                    )}
                  </Box>

                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography sx={{ fontSize: "0.72rem", color: "text.secondary", mb: 0.3 }}>
                      {currentSample ? "测试集参考标签" : "素材来源"}
                    </Typography>
                    <Typography noWrap sx={{ fontSize: "0.92rem", fontWeight: 600, mb: 0.75 }}>
                      {currentSample ? currentSample.gold : imageUri ? imageName : "自定义文本请求"}
                    </Typography>

                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 1.25 }}>
                      {currentSample?.is_disagree && (
                        <Box
                          component="span"
                          sx={{
                            px: 1,
                            py: 0.25,
                            borderRadius: "6px",
                            fontSize: "0.7rem",
                            fontWeight: 500,
                            bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(168,199,250,0.14)" : "#e8f0fe"),
                            color: "primary.main",
                          }}
                        >
                          旧版误判 · SOTA 修正样本
                        </Box>
                      )}
                      {currentSample && !currentSample.has_copy && (
                        <Box
                          component="span"
                          sx={{
                            px: 1,
                            py: 0.25,
                            borderRadius: "6px",
                            fontSize: "0.7rem",
                            fontWeight: 500,
                            bgcolor: (t) => (t.palette.mode === "dark" ? "#282a2c" : "#eef2f6"),
                            color: "text.secondary",
                          }}
                        >
                          无标题正文纯海报
                        </Box>
                      )}
                      {modalityMode !== "full" && (
                        <Box
                          component="span"
                          sx={{
                            px: 1,
                            py: 0.25,
                            borderRadius: "6px",
                            fontSize: "0.7rem",
                            fontWeight: 500,
                            bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(253,214,99,0.15)" : "#fef7e0"),
                            color: (t) => (t.palette.mode === "dark" ? "#fdd663" : "#b06000"),
                          }}
                        >
                          {modalityMode === "pure_image" ? "已清空全部文本" : modalityMode === "raw_ad" ? "已剥离 OCR 与描述" : "自定义编辑"}
                        </Box>
                      )}
                    </Box>

                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleUploadImage} />
                      <Button
                        size="small"
                        variant="text"
                        startIcon={<FileUploadOutlined sx={{ fontSize: 15 }} />}
                        onClick={() => fileInputRef.current?.click()}
                        sx={{ px: 1, py: 0.25, minWidth: 0, fontSize: "0.75rem", color: "text.secondary" }}
                      >
                        上传图片
                      </Button>
                      {imageUri && (
                        <Button
                          size="small"
                          variant="text"
                          startIcon={<DeleteOutlineRounded sx={{ fontSize: 15 }} />}
                          onClick={() => {
                            setImageUri(null);
                            setImageName("");
                          }}
                          sx={{ px: 1, py: 0.25, minWidth: 0, fontSize: "0.75rem", color: "text.secondary" }}
                        >
                          移除
                        </Button>
                      )}
                    </Box>
                  </Box>
                </Box>
              </Box>

              {/* State & Questions Clean Split Editors */}
              <MinimalJsonEditor
                label="State"
                metaText={stateFieldCount}
                value={stateText}
                onChange={(v) => updateStateEditor(v, "custom")}
                error={stateErr}
                sx={{ flex: "0 1 36%", minHeight: 120 }}
                rightActions={
                  <Button
                    size="small"
                    variant="text"
                    onClick={formatJsonEditors}
                    sx={{ px: 1, py: 0, minWidth: 0, fontSize: "0.73rem", color: "text.secondary" }}
                  >
                    格式化
                  </Button>
                }
              />

              <MinimalJsonEditor
                label="Questions"
                metaText={questionCountText}
                value={questionsText}
                onChange={updateQuestionsEditor}
                error={questionsErr}
                sx={{ flex: "1 1 64%", minHeight: 160 }}
                rightActions={
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <Button
                      size="small"
                      variant="text"
                      startIcon={<AddRounded sx={{ fontSize: 14 }} />}
                      onClick={() => addQuestionPreset("noul")}
                      sx={{ px: 1, py: 0, minWidth: 0, fontSize: "0.73rem", color: "text.secondary", whiteSpace: "nowrap" }}
                    >
                      布尔判定
                    </Button>
                    <Button
                      size="small"
                      variant="text"
                      startIcon={<AddRounded sx={{ fontSize: 14 }} />}
                      onClick={() => addQuestionPreset("score")}
                      sx={{ px: 1, py: 0, minWidth: 0, fontSize: "0.73rem", color: "text.secondary", whiteSpace: "nowrap" }}
                    >
                      等级评分
                    </Button>
                  </Box>
                }
              />
            </Box>

            {/* Right Workspace Column: Output */}
            <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, position: "relative" }}>
              {/* M3 Underline Tabs */}
              <Box
                sx={{
                  height: 44,
                  px: 2,
                  borderBottom: 1,
                  borderColor: "divider",
                  display: "flex",
                  alignItems: "center",
                  flexShrink: 0,
                }}
              >
                <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ minHeight: 44 }}>
                  <Tab label="推理结果" />
                  <Tab label={`模型对比 (${catalog.length || 10})`} />
                  <Tab label="Token 视图" />
                </Tabs>

                <Box sx={{ flex: 1 }} />

                {activeTab === 0 && singleResult?.body?.usage && (
                  <Typography
                    sx={{
                      fontSize: "0.74rem",
                      color: "text.secondary",
                      fontFamily: '"Google Sans Mono", Menlo, monospace',
                      mr: 1,
                      display: { xs: "none", lg: "block" },
                    }}
                  >
                    {singleResult.body.usage.input_tokens.toLocaleString()} tok · {singleResult.body.evaluation_time_ms} ms
                  </Typography>
                )}

                <Tooltip title="复制 JSON">
                  <IconButton
                    size="small"
                    onClick={() => {
                      const payload = activeTab === 2 ? promptResult : activeTab === 1 ? compareResult : singleResult?.body;
                      if (!payload) return;
                      navigator.clipboard?.writeText(JSON.stringify(payload, null, 2));
                      setToast("已复制 JSON");
                    }}
                    sx={{ color: "text.secondary" }}
                  >
                    <ContentCopyRounded sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              </Box>

              {isBusy && (
                <LinearProgress
                  sx={{
                    position: "absolute",
                    top: 43,
                    left: 0,
                    right: 0,
                    height: 2,
                    zIndex: 2,
                  }}
                />
              )}

              {/* Output Scroll Content */}
              <Box sx={{ flex: 1, overflow: "auto", p: 2.5 }}>
                {/* TAB 0: Single Model Output */}
                {activeTab === 0 && (
                  <>
                    {!singleResult ? (
                      <Box sx={{ py: 8, textAlign: "center", color: "text.secondary" }}>
                        <Typography sx={{ fontSize: "0.85rem" }}>
                          {loadingSingle ? "正在执行推理…" : "点击右上角「运行」查看模型输出"}
                        </Typography>
                      </Box>
                    ) : (
                      <>
                        {/* Quiet Summary Header Card */}
                        {singleIndustryAns && (
                          <Box
                            sx={{
                              p: 2,
                              mb: 2.75,
                              borderRadius: "12px",
                              bgcolor: (t) => (t.palette.mode === "dark" ? "#26282b" : "#f8fafd"),
                              border: 1,
                              borderColor: "divider",
                              display: "flex",
                              alignItems: "flex-start",
                              justifyContent: "space-between",
                              gap: 2,
                            }}
                          >
                            <Box>
                              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                                {singleIsCorrect !== null && (
                                  <Box
                                    component="span"
                                    sx={{
                                      display: "inline-flex",
                                      alignItems: "center",
                                      gap: 0.4,
                                      px: 1,
                                      py: 0.25,
                                      borderRadius: 999,
                                      fontSize: "0.72rem",
                                      fontWeight: 600,
                                      bgcolor: (t) =>
                                        singleIsCorrect
                                          ? t.palette.mode === "dark"
                                            ? "rgba(129,201,149,0.16)"
                                            : "#e6f4ea"
                                          : t.palette.mode === "dark"
                                          ? "rgba(242,139,130,0.16)"
                                          : "#fce8e6",
                                      color: (t) =>
                                        singleIsCorrect
                                          ? t.palette.mode === "dark"
                                            ? "#81c995"
                                            : "#146c2e"
                                          : t.palette.mode === "dark"
                                          ? "#f28b82"
                                          : "#b3261e",
                                    }}
                                  >
                                    {singleIsCorrect ? <CheckRounded sx={{ fontSize: 13 }} /> : <CloseRounded sx={{ fontSize: 13 }} />}
                                    {singleIsCorrect ? "预测一致" : "预测偏差"}
                                  </Box>
                                )}
                                <Typography sx={{ fontSize: "1.05rem", fontWeight: 600 }}>
                                  {singleIndustryAns.choice}
                                </Typography>
                              </Box>

                              <Typography sx={{ fontSize: "0.78rem", color: "text.secondary" }}>
                                {singleIsCorrect === false && gold ? (
                                  <>
                                    测试集参考标签：<strong>{gold}</strong>
                                    {singleIndustryAns.probabilities?.[gold] != null
                                      ? `（当前模型赋予概率 ${(singleIndustryAns.probabilities[gold] * 100).toFixed(1)}%）`
                                      : ""}
                                  </>
                                ) : (
                                  <>
                                    首选概率 {((singleIndustryAns.probabilities?.[singleIndustryAns.choice] || 0) * 100).toFixed(1)}%
                                    {singleResult.body?.method_meta
                                      ? ` · 校准温度 T = ${singleResult.body.method_meta.temperature_applied} · 视角 ${singleResult.body.method_meta.tta_views_used}×`
                                      : ""}
                                  </>
                                )}
                              </Typography>
                            </Box>

                            <Typography sx={{ fontSize: "0.75rem", color: "text.secondary", textAlign: "right", flexShrink: 0 }}>
                              {MODEL_META[singleResult.body?.model]?.name || singleResult.body?.model}
                            </Typography>
                          </Box>
                        )}

                        {singleResult.body?.answers &&
                          Object.entries(singleResult.body.answers).map(([qid, ans]) => (
                            <ProbabilityDistribution
                              key={qid}
                              qid={qid}
                              ans={ans}
                              gold={qid === "industry" ? gold : null}
                            />
                          ))}

                        {/* Collapsible Raw JSON */}
                        <Box sx={{ borderTop: 1, borderColor: "divider", pt: 1.5 }}>
                          <Button
                            size="small"
                            variant="text"
                            startIcon={rawJsonOpen ? <KeyboardArrowDownRounded /> : <KeyboardArrowRightRounded />}
                            onClick={() => setRawJsonOpen((v) => !v)}
                            sx={{ px: 1, color: "text.secondary", fontSize: "0.76rem" }}
                          >
                            原始 System One 响应 JSON
                          </Button>
                          <Collapse in={rawJsonOpen}>
                            <Box
                              component="pre"
                              sx={{
                                mt: 1,
                                mb: 0,
                                p: 1.75,
                                borderRadius: "10px",
                                overflow: "auto",
                                fontSize: "12px",
                                lineHeight: 1.55,
                                fontFamily: '"Google Sans Mono", "SF Mono", Menlo, monospace',
                                bgcolor: (t) => (t.palette.mode === "dark" ? "#141517" : "#f8fafd"),
                                border: 1,
                                borderColor: "divider",
                              }}
                            >
                              {JSON.stringify(singleResult.body, null, 2)}
                            </Box>
                          </Collapse>
                        </Box>
                      </>
                    )}
                  </>
                )}

                {/* TAB 1: All-Methods Benchmark */}
                {activeTab === 1 && (
                  <>
                    {!compareResult ? (
                      <Box sx={{ py: 8, textAlign: "center" }}>
                        <Typography sx={{ fontSize: "0.85rem", color: "text.secondary", mb: 2 }}>
                          {loadingCompare
                            ? "正在并行评测全部已加载模型…"
                            : "在同一输入下并行对比全部 10 种模型的预测结果、校准概率与 Token 开销"}
                        </Typography>
                        {!loadingCompare && (
                          <Button variant="contained" size="small" onClick={runCompareAll}>
                            开始全模型对比
                          </Button>
                        )}
                      </Box>
                    ) : (
                      <>
                        {/* Unified Google Metric Strip */}
                        {compareKpis && (
                          <Box
                            sx={{
                              display: "grid",
                              gridTemplateColumns: "repeat(4, 1fr)",
                              borderRadius: "12px",
                              bgcolor: (t) => (t.palette.mode === "dark" ? "#26282b" : "#f8fafd"),
                              border: 1,
                              borderColor: "divider",
                              mb: 2,
                              overflow: "hidden",
                            }}
                          >
                            {[
                              {
                                k: "命中模型",
                                v: compareKpis.correctCount != null ? `${compareKpis.correctCount} / ${compareKpis.total}` : `${compareKpis.total}`,
                                sub: compareKpis.gold ? `参考：${compareKpis.gold}` : "无参考标签",
                              },
                              {
                                k: "4B 视觉增益",
                                v:
                                  compareKpis.pGoldSota4b != null && compareKpis.pGoldText4b != null
                                    ? `${((compareKpis.pGoldSota4b - compareKpis.pGoldText4b) * 100 >= 0 ? "+" : "") + ((compareKpis.pGoldSota4b - compareKpis.pGoldText4b) * 100).toFixed(1)} pp`
                                    : "—",
                                sub:
                                  compareKpis.pGoldSota4b != null && compareKpis.pGoldText4b != null
                                    ? `${(compareKpis.pGoldSota4b * 100).toFixed(1)}% vs ${(compareKpis.pGoldText4b * 100).toFixed(1)}%`
                                    : "目标类概率",
                              },
                              {
                                k: "Prompt 压缩",
                                v:
                                  compareKpis.tokV3 && compareKpis.tokV1
                                    ? `-${(((compareKpis.tokV1 - compareKpis.tokV3) / compareKpis.tokV1) * 100).toFixed(1)}%`
                                    : "—",
                                sub: compareKpis.tokV3 && compareKpis.tokV1 ? `${compareKpis.tokV3} vs ${compareKpis.tokV1} tok` : "Token 开销",
                              },
                              {
                                k: "总耗时",
                                v: `${(compareResult.total_wall_ms / 1000).toFixed(2)} s`,
                                sub: `校准${compareResult.calibrated ? "开" : "关"} · ${compareResult.tta_views}× 视角`,
                              },
                            ].map((m, idx) => (
                              <Box
                                key={m.k}
                                sx={{
                                  px: 1.5,
                                  py: 1.25,
                                  minWidth: 0,
                                  borderLeft: idx > 0 ? 1 : 0,
                                  borderColor: "divider",
                                }}
                              >
                                <Typography noWrap sx={{ fontSize: "0.71rem", color: "text.secondary" }}>{m.k}</Typography>
                                <Typography
                                  noWrap
                                  sx={{
                                    fontSize: "1rem",
                                    fontWeight: 600,
                                    my: 0.15,
                                    fontVariantNumeric: "tabular-nums",
                                  }}
                                >
                                  {m.v}
                                </Typography>
                                <Typography noWrap title={m.sub} sx={{ fontSize: "0.69rem", color: "text.secondary" }}>
                                  {m.sub}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        )}

                        <TableContainer sx={{ overflowX: "auto" }}>
                          <Table size="small" sx={{ "& .MuiTableCell-root": { px: 1, py: 0.9 } }}>
                            <TableHead>
                              <TableRow>
                                <TableCell sx={{ pl: "4px !important" }}>模型</TableCell>
                                <TableCell>预测结果</TableCell>
                                <TableCell align="right">首选</TableCell>
                                <TableCell align="right">参考类</TableCell>
                                <TableCell align="right" sx={{ pr: "4px !important" }}>开销 / Token流</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {compareResult.results.map((item) => {
                                const mInfo = MODEL_META[item.model] || { name: item.model, shortName: item.model, spec: "" };
                                if (!item.ok) {
                                  return (
                                    <TableRow key={item.model}>
                                      <TableCell sx={{ whiteSpace: "nowrap", pl: "4px !important" }}>{mInfo.shortName || mInfo.name}</TableCell>
                                      <TableCell colSpan={4} sx={{ color: "error.main" }}>
                                        {item.error}
                                      </TableCell>
                                    </TableRow>
                                  );
                                }
                                const qid = compareKpis?.qid || "industry";
                                const res = item.result;
                                const ans = res.answers?.[qid] || {};
                                const pred =
                                  ans.type === "choice"
                                    ? ans.choice
                                    : ans.type === "noul"
                                    ? `true: ${ans.noul}`
                                    : `score: ${ans.score}`;
                                const probs = ans.probabilities || {};
                                const topP = ans.type === "choice" && probs[ans.choice] != null ? probs[ans.choice] : null;
                                const goldP = gold && probs[gold] != null ? probs[gold] : null;
                                const isOk = gold && ans.type === "choice" ? pred === gold : null;

                                return (
                                  <TableRow key={item.model} hover>
                                    <TableCell sx={{ whiteSpace: "nowrap", pl: "4px !important" }}>
                                      <Typography sx={{ fontSize: "0.79rem", fontWeight: 500, lineHeight: 1.3 }}>
                                        {mInfo.shortName || mInfo.name}
                                      </Typography>
                                      <Typography sx={{ fontSize: "0.68rem", color: "text.secondary", lineHeight: 1.25 }}>
                                        {mInfo.spec}
                                      </Typography>
                                    </TableCell>

                                    <TableCell sx={{ whiteSpace: "nowrap" }}>
                                      <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.75 }}>
                                        {isOk !== null && (
                                          <Box
                                            sx={{
                                              width: 7,
                                              height: 7,
                                              borderRadius: "50%",
                                              flexShrink: 0,
                                              bgcolor: (t) =>
                                                isOk
                                                  ? t.palette.mode === "dark"
                                                    ? "#81c995"
                                                    : "#1e8e3e"
                                                  : t.palette.mode === "dark"
                                                  ? "#f28b82"
                                                  : "#d93025",
                                            }}
                                          />
                                        )}
                                        <Typography
                                          sx={{
                                            fontSize: "0.79rem",
                                            fontWeight: isOk ? 600 : 400,
                                            color: isOk === false ? "text.secondary" : "text.primary",
                                          }}
                                        >
                                          {pred}
                                        </Typography>
                                      </Box>
                                    </TableCell>

                                    <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                                      {topP != null ? (
                                        <Box sx={{ display: "inline-flex", flexDirection: "column", alignItems: "flex-end", width: 48 }}>
                                          <Typography
                                            sx={{
                                              fontSize: "0.75rem",
                                              fontFamily: '"Google Sans Mono", Menlo, monospace',
                                              fontVariantNumeric: "tabular-nums",
                                              fontWeight: 500,
                                            }}
                                          >
                                            {(topP * 100).toFixed(1)}%
                                          </Typography>
                                          <Box
                                            sx={{
                                              width: "100%",
                                              height: 3,
                                              mt: 0.3,
                                              borderRadius: 999,
                                              bgcolor: (t) => (t.palette.mode === "dark" ? "#2c2d30" : "#eef2f6"),
                                            }}
                                          >
                                            <Box
                                              sx={{
                                                height: "100%",
                                                width: `${Math.max(2, topP * 100)}%`,
                                                borderRadius: 999,
                                                bgcolor: (t) =>
                                                  isOk === false
                                                    ? t.palette.mode === "dark"
                                                      ? "#f28b82"
                                                      : "#d93025"
                                                    : "primary.main",
                                              }}
                                            />
                                          </Box>
                                        </Box>
                                      ) : (
                                        "—"
                                      )}
                                    </TableCell>

                                    <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                                      {goldP != null ? (
                                        <Box sx={{ display: "inline-flex", flexDirection: "column", alignItems: "flex-end", width: 48 }}>
                                          <Typography
                                            sx={{
                                              fontSize: "0.75rem",
                                              fontFamily: '"Google Sans Mono", Menlo, monospace',
                                              fontVariantNumeric: "tabular-nums",
                                              color: (t) => (goldP >= 0.5 ? (t.palette.mode === "dark" ? "#81c995" : "#146c2e") : "text.secondary"),
                                              fontWeight: goldP >= 0.5 ? 600 : 400,
                                            }}
                                          >
                                            {(goldP * 100).toFixed(1)}%
                                          </Typography>
                                          <Box
                                            sx={{
                                              width: "100%",
                                              height: 3,
                                              mt: 0.3,
                                              borderRadius: 999,
                                              bgcolor: (t) => (t.palette.mode === "dark" ? "#2c2d30" : "#eef2f6"),
                                            }}
                                          >
                                            <Box
                                              sx={{
                                                height: "100%",
                                                width: `${Math.max(2, goldP * 100)}%`,
                                                borderRadius: 999,
                                                bgcolor: (t) => (t.palette.mode === "dark" ? "#81c995" : "#1e8e3e"),
                                              }}
                                            />
                                          </Box>
                                        </Box>
                                      ) : (
                                        "—"
                                      )}
                                    </TableCell>

                                    <TableCell align="right" sx={{ whiteSpace: "nowrap", pr: "4px !important" }}>
                                      <Box
                                        component="button"
                                        type="button"
                                        onClick={() => fetchPromptOnly(item.model)}
                                        title="点击查看该模型的 Prompt 与 Token 流"
                                        sx={{
                                          border: 0,
                                          bgcolor: "transparent",
                                          p: 0,
                                          cursor: "pointer",
                                          textAlign: "right",
                                          fontFamily: '"Google Sans Mono", Menlo, monospace',
                                          fontSize: "0.72rem",
                                          color: "primary.main",
                                          "&:hover": { textDecoration: "underline" },
                                        }}
                                      >
                                        <div>{res.usage?.input_tokens} tok →</div>
                                        <div style={{ color: theme.palette.text.secondary, fontSize: "0.68rem" }}>
                                          {res.evaluation_time_ms} ms
                                        </div>
                                      </Box>
                                    </TableCell>
                                  </TableRow>
                                );
                              })}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </>
                    )}
                  </>
                )}

                {/* TAB 2: Token & Prompt Stream */}
                {activeTab === 2 && (
                  <>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: 1.5,
                        mb: 2,
                      }}
                    >
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <Typography sx={{ fontSize: "0.78rem", color: "text.secondary" }}>透视模型</Typography>
                        <Select
                          size="small"
                          value={promptResult?.model || selectedModel}
                          onChange={(e) => fetchPromptOnly(e.target.value)}
                          sx={{ height: 32, fontSize: "0.8rem", borderRadius: "8px" }}
                        >
                          {displayCatalog.map((m) => (
                            <MenuItem key={m.id} value={m.id} sx={{ fontSize: "0.8rem" }}>
                              {m.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </Box>

                      {promptResult?.summary && (
                        <Typography sx={{ fontSize: "0.76rem", color: "text.secondary", fontFamily: "monospace" }}>
                          {promptResult.summary.prompt_format} · {promptResult.summary.total_tokens?.toLocaleString()} tokens（图像{" "}
                          {promptResult.summary.image_tokens}）
                          {promptResult.rendered?.image?.grid_patches_thw
                            ? ` · 网格 ${promptResult.rendered.image.grid_patches_thw.join("×")}`
                            : ""}
                        </Typography>
                      )}
                    </Box>

                    {!promptResult ? (
                      <Box sx={{ py: 6, textAlign: "center" }}>
                        <Button variant="outlined" size="small" onClick={() => fetchPromptOnly(selectedModel)}>
                          加载 Token 视图
                        </Button>
                      </Box>
                    ) : (
                      <>
                        {(() => {
                          const groupedSegs = [];
                          for (const g of promptResult.segments || []) {
                            if (!groupedSegs.length || groupedSegs[groupedSegs.length - 1].name !== g.segment) {
                              groupedSegs.push({ name: g.segment, items: [] });
                            }
                            groupedSegs[groupedSegs.length - 1].items.push(g);
                          }
                          return groupedSegs.map((sg, idx) => (
                            <Box key={idx} sx={{ mb: 2.25 }}>
                              <Typography
                                sx={{
                                  fontSize: "0.72rem",
                                  fontWeight: 600,
                                  color: "text.secondary",
                                  fontFamily: "monospace",
                                  mb: 0.75,
                                }}
                              >
                                {sg.name}
                              </Typography>
                              <Box
                                sx={{
                                  p: 1.75,
                                  borderRadius: "10px",
                                  bgcolor: (t) => (t.palette.mode === "dark" ? "#141517" : "#f8fafd"),
                                  border: 1,
                                  borderColor: "divider",
                                  fontFamily: '"Google Sans Mono", "SF Mono", Menlo, monospace',
                                  fontSize: "12px",
                                  lineHeight: 1.75,
                                  whiteSpace: "pre-wrap",
                                  wordBreak: "break-word",
                                }}
                              >
                                {sg.items.map((it, k) => {
                                  if (it.kind === "special") {
                                    return (
                                      <Box
                                        key={k}
                                        component="span"
                                        sx={{
                                          display: "inline-block",
                                          px: 0.7,
                                          py: "1px",
                                          mx: 0.25,
                                          my: "1px",
                                          borderRadius: "4px",
                                          fontSize: "11px",
                                          bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(168,199,250,0.14)" : "#e8f0fe"),
                                          color: "primary.main",
                                        }}
                                      >
                                        {it.text}
                                      </Box>
                                    );
                                  }
                                  if (it.kind === "image") {
                                    return (
                                      <Box
                                        key={k}
                                        component="span"
                                        sx={{
                                          display: "inline-block",
                                          px: 0.9,
                                          py: "1px",
                                          mx: 0.35,
                                          my: "1px",
                                          borderRadius: "4px",
                                          fontSize: "11px",
                                          fontWeight: 600,
                                          bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(129,201,149,0.16)" : "#e6f4ea"),
                                          color: (t) => (t.palette.mode === "dark" ? "#81c995" : "#146c2e"),
                                        }}
                                      >
                                        {it.text} × {it.count}
                                      </Box>
                                    );
                                  }
                                  return <span key={k}>{it.text}</span>;
                                })}
                              </Box>
                            </Box>
                          ));
                        })()}

                        <Box sx={{ borderTop: 1, borderColor: "divider", pt: 1.5 }}>
                          <Button
                            size="small"
                            variant="text"
                            startIcon={tokenTableOpen ? <KeyboardArrowDownRounded /> : <KeyboardArrowRightRounded />}
                            onClick={() => setTokenTableOpen((v) => !v)}
                            sx={{ px: 1, color: "text.secondary", fontSize: "0.76rem" }}
                          >
                            词表 Token 映射明细（{promptResult.tokens?.length || 0}）
                          </Button>
                          <Collapse in={tokenTableOpen}>
                            <TableContainer sx={{ maxHeight: 360, mt: 1 }}>
                              <Table size="small" stickyHeader>
                                <TableHead>
                                  <TableRow>
                                    <TableCell>#</TableCell>
                                    <TableCell>Token ID</TableCell>
                                    <TableCell>Piece</TableCell>
                                    <TableCell>段落</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {(showAllTokens ? promptResult.tokens : (promptResult.tokens || []).slice(0, 200)).map(
                                    (tok) => (
                                      <TableRow key={tok.i} hover>
                                        <TableCell sx={{ fontFamily: "monospace", color: "text.secondary" }}>{tok.i}</TableCell>
                                        <TableCell sx={{ fontFamily: "monospace" }}>{tok.id}</TableCell>
                                        <TableCell sx={{ fontFamily: "monospace" }}>{tok.piece}</TableCell>
                                        <TableCell sx={{ color: "text.secondary" }}>{tok.segment}</TableCell>
                                      </TableRow>
                                    )
                                  )}
                                </TableBody>
                              </Table>
                            </TableContainer>
                            {(promptResult.tokens?.length || 0) > 200 && !showAllTokens && (
                              <Box sx={{ pt: 1 }}>
                                <Button size="small" variant="text" onClick={() => setShowAllTokens(true)}>
                                  显示其余 {promptResult.tokens.length - 200} 项
                                </Button>
                              </Box>
                            )}
                          </Collapse>
                        </Box>
                      </>
                    )}
                  </>
                )}
              </Box>
            </Box>
          </Box>

          {/* Right Google AI Studio `Run settings` Rail */}
          {settingsOpen && (
            <Box
              component="aside"
              sx={{
                width: 252,
                flexShrink: 0,
                display: { xs: "none", md: "flex" },
                flexDirection: "column",
                gap: 2.25,
                px: 1,
                py: 0.75,
                overflow: "auto",
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <Typography sx={{ fontSize: "0.825rem", fontWeight: 600 }}>运行配置</Typography>
                <IconButton size="small" onClick={() => setSettingsOpen(false)} sx={{ width: 26, height: 26 }}>
                  <CloseRounded sx={{ fontSize: 16 }} />
                </IconButton>
              </Box>

              {/* Model Selector */}
              <Box>
                <Typography sx={{ fontSize: "0.74rem", fontWeight: 500, color: "text.secondary", mb: 0.75 }}>
                  模型选择
                </Typography>
                <Select
                  fullWidth
                  size="small"
                  value={selectedModel}
                  renderValue={(val) => MODEL_META[val]?.name || val}
                  onChange={(e) => {
                    const next = e.target.value;
                    setSelectedModel(next);
                    runSingleModel(next);
                  }}
                  sx={{
                    borderRadius: "10px",
                    bgcolor: "background.paper",
                    fontSize: "0.8125rem",
                    fontWeight: 500,
                    "& .MuiOutlinedInput-notchedOutline": { borderColor: "divider" },
                  }}
                >
                  {groupedCatalog.map(([groupName, items]) => [
                    <ListSubheader
                      key={`g-${groupName}`}
                      sx={{
                        lineHeight: "28px",
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        color: "text.secondary",
                        bgcolor: "background.paper",
                      }}
                    >
                      {groupName}
                    </ListSubheader>,
                    ...items.map((m) => (
                      <MenuItem key={m.id} value={m.id} sx={{ py: 0.85 }}>
                        <Box>
                          <Typography sx={{ fontSize: "0.8125rem", fontWeight: 500 }}>{m.name}</Typography>
                          <Typography sx={{ fontSize: "0.7rem", color: "text.secondary" }}>
                            {m.spec} · {m.metric}
                          </Typography>
                        </Box>
                      </MenuItem>
                    )),
                  ])}
                </Select>

                {/* Quiet Model Spec Readout */}
                <Box
                  sx={{
                    mt: 1,
                    p: 1.25,
                    borderRadius: "10px",
                    bgcolor: (t) => (t.palette.mode === "dark" ? "#1e1f20" : "#eef2f7"),
                  }}
                >
                  <Typography sx={{ fontSize: "0.73rem", color: "text.primary", fontWeight: 500 }}>
                    {currentModelMeta.spec}
                  </Typography>
                  <Typography sx={{ fontSize: "0.71rem", color: "text.secondary", mt: 0.25 }}>
                    {currentModelMeta.metric}
                  </Typography>
                </Box>
              </Box>

              <Divider />

              {/* Inference Switches */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.75 }}>
                <Typography sx={{ fontSize: "0.74rem", fontWeight: 500, color: "text.secondary" }}>
                  校准与测试时增强
                </Typography>

                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
                  <Box>
                    <Typography sx={{ fontSize: "0.8rem", fontWeight: 500 }}>温度校准 (Locked T*)</Typography>
                    <Typography sx={{ fontSize: "0.7rem", color: "text.secondary" }}>
                      验证集锁定温度，抑制过自信
                    </Typography>
                  </Box>
                  <M3Switch checked={calibrated} onChange={(e) => setCalibrated(e.target.checked)} />
                </Box>

                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
                  <Box>
                    <Typography sx={{ fontSize: "0.8rem", fontWeight: 500 }}>循环移位集成 (TTA-5)</Typography>
                    <Typography sx={{ fontSize: "0.7rem", color: "text.secondary" }}>
                      5 视角融合消除选项位置偏置
                    </Typography>
                  </Box>
                  <M3Switch checked={tta5} onChange={(e) => setTta5(e.target.checked)} />
                </Box>
              </Box>

              <Divider />

              {/* Image Field Position */}
              <Box>
                <Typography sx={{ fontSize: "0.74rem", fontWeight: 500, color: "text.secondary", mb: 0.85 }}>
                  图像字段位置
                </Typography>
                <SegmentedControl
                  fullWidth
                  value={imgPos}
                  onChange={setImgPos}
                  options={[
                    { value: "start", label: "开头" },
                    { value: "middle", label: "中间" },
                    { value: "end", label: "结尾" },
                  ]}
                />
              </Box>

              <Divider />

              {/* Quiet Usage Note */}
              <Box>
                <Typography sx={{ fontSize: "0.74rem", fontWeight: 500, color: "text.secondary", mb: 0.6 }}>
                  实验建议
                </Typography>
                <Typography sx={{ fontSize: "0.73rem", color: "text.secondary", lineHeight: 1.6 }}>
                  将输入切换至「纯图像」并运行「全模型对比」，可直观观察纯文本模型退化与多模态 SOTA 直接识别海报像素的差异。
                </Typography>
              </Box>
            </Box>
          )}
        </Box>

        {/* Clean Lightbox Modal */}
        <Dialog open={lightboxOpen} onClose={() => setLightboxOpen(false)} maxWidth="lg">
          <DialogContent sx={{ position: "relative", bgcolor: "#131314", p: 2.5, textAlign: "center" }}>
            <IconButton
              size="small"
              onClick={() => setLightboxOpen(false)}
              sx={{ position: "absolute", top: 10, right: 10, color: "#e3e3e3", bgcolor: "rgba(255,255,255,0.1)" }}
            >
              <CloseRounded fontSize="small" />
            </IconButton>
            {imageUri && (
              <Box
                component="img"
                src={imageUri}
                alt="广告原图"
                sx={{ maxWidth: "86vw", maxHeight: "82vh", objectFit: "contain", borderRadius: "8px" }}
              />
            )}
          </DialogContent>
        </Dialog>

        <Snackbar
          open={Boolean(toast)}
          autoHideDuration={2600}
          onClose={() => setToast("")}
          message={toast}
          anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        />
      </Box>
    </ThemeProvider>
  );
}
