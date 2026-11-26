import React, { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Paper,
  Card,
  CardContent,
  Button,
  IconButton,
  Select,
  MenuItem,
  ListSubheader,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Chip,
  Stack,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Alert,
  AlertTitle,
  Tooltip,
  Dialog,
  DialogContent,
  DialogTitle,
  Snackbar,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Avatar,
  alpha,
} from "@mui/material";
import {
  AutoAwesomeRounded,
  BoltRounded,
  PlayArrowRounded,
  ShuffleRounded,
  DarkModeRounded,
  LightModeRounded,
  UploadFileRounded,
  DeleteOutlineRounded,
  ContentCopyRounded,
  ExpandMoreRounded,
  ZoomInRounded,
  ImageRounded,
  VisibilityOffRounded,
  ContentCutRounded,
  RestoreRounded,
  CheckCircleRounded,
  CancelRounded,
  AddCircleOutlineRounded,
  CodeRounded,
  DataObjectRounded,
  PsychologyRounded,
  SpeedRounded,
  TokenRounded,
  StarRounded,
  CloseRounded,
} from "@mui/icons-material";

function buildTheme(mode) {
  const isDark = mode === "dark";
  return createTheme({
    palette: {
      mode,
      primary: {
        main: isDark ? "#60a5fa" : "#2563eb",
        light: isDark ? "#93c5fd" : "#60a5fa",
        dark: isDark ? "#3b82f6" : "#1d4ed8",
      },
      secondary: {
        main: isDark ? "#a78bfa" : "#7c3aed",
      },
      success: {
        main: isDark ? "#34d399" : "#16a34a",
      },
      warning: {
        main: isDark ? "#fbbf24" : "#d97706",
      },
      error: {
        main: isDark ? "#f87171" : "#dc2626",
      },
      background: {
        default: isDark ? "#0b0f19" : "#f4f6fa",
        paper: isDark ? "#111827" : "#ffffff",
      },
      divider: isDark ? "rgba(255,255,255,0.09)" : "rgba(15,23,42,0.09)",
    },
    shape: { borderRadius: 10 },
    typography: {
      fontFamily:
        'Inter, -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
      button: { textTransform: "none", fontWeight: 600 },
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: { backgroundImage: "none" },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: { borderRadius: 8, boxShadow: "none", "&:hover": { boxShadow: "none" } },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: { fontWeight: 500 },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          head: {
            fontWeight: 700,
            fontSize: "0.76rem",
            letterSpacing: "0.02em",
            whiteSpace: "nowrap",
          },
          body: { fontSize: "0.8125rem" },
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

function CodeEditorCard({ title, icon, value, onChange, error, actions, minHeight = 165 }) {
  const gutterRef = useRef(null);
  const lineCount = useMemo(() => (value.match(/\n/g) || []).length + 1, [value]);
  const lineNumbers = useMemo(
    () => Array.from({ length: lineCount }, (_, i) => i + 1).join("\n"),
    [lineCount]
  );

  const handleScroll = (e) => {
    if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop;
  };

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
    <Paper variant="outlined" sx={{ display: "flex", flexDirection: "column", overflow: "hidden", flex: 1, minHeight }}>
      <Box
        sx={{
          px: 1.5,
          py: 0.75,
          display: "flex",
          alignItems: "center",
          gap: 1,
          borderBottom: 1,
          borderColor: "divider",
          bgcolor: (t) => alpha(t.palette.primary.main, t.palette.mode === "dark" ? 0.06 : 0.03),
        }}
      >
        {icon}
        <Typography variant="subtitle2" sx={{ fontWeight: 700, fontSize: "0.8rem" }}>
          {title}
        </Typography>
        {error ? (
          <Chip size="small" color="error" variant="outlined" label="JSON 语法错误" sx={{ height: 20, fontSize: "0.7rem" }} />
        ) : (
          <Chip size="small" color="success" variant="outlined" label="JSON 合法" sx={{ height: 20, fontSize: "0.7rem" }} />
        )}
        <Box sx={{ flex: 1 }} />
        {actions}
      </Box>
      <Box sx={{ display: "flex", flex: 1, minHeight: 0, position: "relative" }}>
        <Box
          ref={gutterRef}
          component="pre"
          sx={{
            m: 0,
            py: 1.25,
            px: 1,
            width: 38,
            textAlign: "right",
            userSelect: "none",
            overflow: "hidden",
            fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: "12px",
            lineHeight: "1.55",
            color: "text.disabled",
            bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(255,255,255,0.02)" : "rgba(15,23,42,0.025)"),
            borderRight: 1,
            borderColor: "divider",
          }}
        >
          {lineNumbers}
        </Box>
        <Box
          component="textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onScroll={handleScroll}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          sx={{
            flex: 1,
            width: "100%",
            border: 0,
            outline: "none",
            resize: "none",
            p: 1.25,
            m: 0,
            bgcolor: "transparent",
            color: "text.primary",
            fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: "12.5px",
            lineHeight: "1.55",
            overflow: "auto",
          }}
        />
      </Box>
      {error && (
        <Box sx={{ px: 1.5, py: 0.5, bgcolor: (t) => alpha(t.palette.error.main, 0.1), borderTop: 1, borderColor: "error.main" }}>
          <Typography variant="caption" color="error.main" sx={{ fontFamily: "monospace" }}>
            {error}
          </Typography>
        </Box>
      )}
    </Paper>
  );
}

function QuestionDistributionCard({ qid, ans, gold }) {
  if (!ans) return null;

  if (ans.type === "choice") {
    const entries = Object.entries(ans.probabilities || {}).sort((a, b) => b[1] - a[1]);
    const topEntries = entries.slice(0, 8);
    return (
      <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5, flexWrap: "wrap", useFlexGap: true }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            决策项 `{qid}` · Top-8 候选概率分布
          </Typography>
          <Chip size="small" color="primary" label={`首选：${ans.choice}`} />
          {ans.confidence != null && (
            <Chip size="small" variant="outlined" label={`归一化置信度 ${(ans.confidence * 100).toFixed(1)}%`} />
          )}
          <Box sx={{ flex: 1 }} />
          <Typography variant="caption" color="text.secondary">
            共 {entries.length} 个候选类目
          </Typography>
        </Stack>
        <Stack spacing={1}>
          {topEntries.map(([label, prob], idx) => {
            const isGold = gold && label === gold;
            const isPredWrong = gold && label === ans.choice && label !== gold;
            const barColor = isGold ? "success" : isPredWrong ? "error" : idx === 0 ? "primary" : "inherit";
            return (
              <Box key={label} sx={{ display: "grid", gridTemplateColumns: "minmax(140px, 210px) 1fr 64px", alignItems: "center", gap: 1.5 }}>
                <Stack direction="row" alignItems="center" spacing={0.5} sx={{ minWidth: 0 }}>
                  {isGold && <StarRounded fontSize="small" color="success" sx={{ fontSize: 16, flexShrink: 0 }} />}
                  <Typography
                    variant="body2"
                    noWrap
                    title={label}
                    sx={{
                      fontWeight: isGold || idx === 0 ? 700 : 400,
                      color: isGold ? "success.main" : isPredWrong ? "error.main" : "text.primary",
                      fontSize: "0.8rem",
                    }}
                  >
                    {label}
                  </Typography>
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, Math.max(0.8, prob * 100))}
                  color={barColor === "inherit" ? "primary" : barColor}
                  sx={{
                    height: 9,
                    borderRadius: 99,
                    opacity: barColor === "inherit" ? 0.55 : 1,
                    bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(255,255,255,0.07)" : "rgba(15,23,42,0.06)"),
                  }}
                />
                <Typography
                  variant="body2"
                   sx={{
                    textAlign: "right",
                    fontVariantNumeric: "tabular-nums",
                    fontWeight: idx === 0 || isGold ? 700 : 500,
                    fontSize: "0.8rem",
                  }}
                >
                  {(prob * 100).toFixed(1)}%
                </Typography>
              </Box>
            );
          })}
        </Stack>
      </Paper>
    );
  }

  if (ans.type === "noul") {
    const pTrue = ans.probabilities?.true ?? ans.noul ?? 0;
    const pFalse = ans.probabilities?.false ?? 1 - pTrue;
    return (
      <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            布尔判定 `{qid}` (noul)
          </Typography>
          <Chip
            size="small"
            color={pTrue >= 0.5 ? "warning" : "success"}
            label={`P(true) = ${(pTrue * 100).toFixed(1)}%`}
          />
        </Stack>
        <Stack spacing={1}>
          {[
            ["true (命中)", pTrue, "warning"],
            ["false (未命中)", pFalse, "success"],
          ].map(([k, p, col]) => (
            <Box key={k} sx={{ display: "grid", gridTemplateColumns: "150px 1fr 64px", alignItems: "center", gap: 1.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 600, fontSize: "0.8rem" }}>{k}</Typography>
              <LinearProgress variant="determinate" value={Math.max(1, p * 100)} color={col} sx={{ height: 9, borderRadius: 99 }} />
              <Typography variant="body2" sx={{ textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
                {(p * 100).toFixed(1)}%
              </Typography>
            </Box>
          ))}
        </Stack>
      </Paper>
    );
  }

  // score type
  const probs = Object.entries(ans.probabilities || {});
  return (
    <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          等级评分 `{qid}` (score)
        </Typography>
        <Chip size="small" color="secondary" label={`期望得分：${ans.score}`} />
        {ans.confidence != null && <Chip size="small" variant="outlined" label={`集中度 ${(ans.confidence * 100).toFixed(1)}%`} />}
      </Stack>
      <Stack spacing={1}>
        {probs.map(([k, p]) => (
          <Box key={k} sx={{ display: "grid", gridTemplateColumns: "200px 1fr 64px", alignItems: "center", gap: 1.5 }}>
            <Typography variant="body2" noWrap title={ans.legend?.[k] || k} sx={{ fontSize: "0.8rem" }}>
              [{k}] {ans.legend?.[k] || k}
            </Typography>
            <LinearProgress variant="determinate" value={Math.max(1, p * 100)} color="secondary" sx={{ height: 9, borderRadius: 99 }} />
            <Typography variant="body2" sx={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
              {(p * 100).toFixed(1)}%
            </Typography>
          </Box>
        ))}
      </Stack>
    </Paper>
  );
}

export default function App() {
  const [themeMode, setThemeMode] = useState("light");
  const theme = useMemo(() => buildTheme(themeMode), [themeMode]);

  const [catalog, setCatalog] = useState([]);
  const [sampleCounts, setSampleCounts] = useState({ all: 150, disagree: 25, poster_only: 33 });
  const [sampleMode, setSampleMode] = useState("disagree");
  const [selectedModel, setSelectedModel] = useState("nimble-4b-mm-sota");

  const [currentSample, setCurrentSample] = useState(null);
  const [savedFullState, setSavedFullState] = useState(null);
  const [modalityMode, setModalityMode] = useState("full"); // full | raw_ad | pure_image | custom

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

  const [activeTab, setActiveTab] = useState(0); // 0: single, 1: compare, 2: prompt
  const [loadingSingle, setLoadingSingle] = useState(false);
  const [loadingCompare, setLoadingCompare] = useState(false);
  const [loadingPrompt, setLoadingPrompt] = useState(false);

  const [singleResult, setSingleResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [promptResult, setPromptResult] = useState(null);
  const [showAllTokens, setShowAllTokens] = useState(false);
  const [toast, setToast] = useState("");

  const fileInputRef = useRef(null);

  const groupedCatalog = useMemo(() => {
    const groups = new Map();
    for (const m of catalog) {
      const g = m.group || "Models";
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(m);
    }
    return Array.from(groups.entries());
  }, [catalog]);

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

  const fetchRandomSample = useCallback(
    async (modeOverride) => {
      const m = modeOverride ?? sampleMode;
      try {
        const res = await fetch(`/api/sample/random?mode=${encodeURIComponent(m)}`);
        const s = await res.json();
        setCurrentSample(s);
        setSavedFullState(s.state);
        updateStateEditor(s.state, "full");
        updateQuestionsEditor(s.questions);
        setImageUri(s.image);
        setImageName(`${s.id}.jpg`);
        setSingleResult(null);
        setCompareResult(null);
        setPromptResult(null);
        setShowAllTokens(false);
      } catch (e) {
        setToast("样本加载失败: " + String(e));
      }
    },
    [sampleMode, updateStateEditor, updateQuestionsEditor]
  );

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/catalog");
        const cat = await res.json();
        if (cat.models?.length) {
          setCatalog(cat.models);
          setSelectedModel(cat.models[0].id);
        }
        if (cat.sample_counts) setSampleCounts(cat.sample_counts);
        await fetchRandomSample("disagree");
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
      setToast("State JSON 格式错误: " + e.message);
      return null;
    }
    try {
      qObj = JSON.parse(questionsText);
    } catch (e) {
      setToast("Questions JSON 格式错误: " + e.message);
      return null;
    }
    const finalState = imageUri ? withImagePos(stObj, imageUri, imgPos) : stObj;
    return {
      state: finalState,
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
        if (activeTab === 1 && ! modelOverride) setActiveTab(0);
      } catch (e) {
        setToast("单模型推理出错: " + String(e));
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
        else setToast("Prompt 获取失败: " + (jp.detail || rp.statusText));
      } catch (e) {
        setToast("Prompt 请求出错: " + String(e));
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
      const data = await res.json();
      setCompareResult(data);
    } catch (e) {
      setToast("全方法横评失败: " + String(e));
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

  const handleModalitySwitch = (_, nextMode) => {
    if (!nextMode) return;
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
      setToast(`已加载自定义图片 ${f.name}`);
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
          instructions: "该广告画面或文案中是否包含夸大绝对化用语、功效承诺或诱导点击风险？",
          criteria: { true: "包含风险暗示或功效承诺", false: "正常合规客观展示" },
        };
      } else {
        let i = 1;
        while (q[`visual_clarity_${i}`]) i++;
        q[`visual_clarity_${i}`] = {
          type: "score",
          instructions: "评估该广告素材对于目标行业意图表达的清晰度等级",
          criteria: ["0分：意图极度模糊或无关", "1分：仅有弱相关线索", "2分：行业主体清晰明确", "3分：品牌、产品与行业高度一目了然"],
        };
      }
      updateQuestionsEditor(q);
    } catch {
      setToast("请先修复 Questions JSON 语法后再添加题目");
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

  const copyCurrentJson = () => {
    const payload =
      activeTab === 2
        ? promptResult
        : activeTab === 1
        ? compareResult
        : singleResult?.body;
    if (!payload) return;
    navigator.clipboard?.writeText(JSON.stringify(payload, null, 2));
    setToast("已复制 JSON 到剪贴板");
  };

  // Compare KPI metrics
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
      <Box sx={{ height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Top Material AppBar */}
        <AppBar
          position="static"
          color="default"
          elevation={0}
          sx={{
            borderBottom: 1,
            borderColor: "divider",
            bgcolor: "background.paper",
          }}
        >
          <Toolbar variant="dense" sx={{ gap: 1.25, py: 0.75, flexWrap: "wrap" }}>
            <Avatar
              sx={{
                width: 32,
                height: 32,
                bgcolor: "primary.main",
                color: "#fff",
              }}
            >
              <AutoAwesomeRounded sx={{ fontSize: 19 }} />
            </Avatar>
            <Box>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography variant="subtitle1" sx={{ fontWeight: 800, letterSpacing: "-0.01em", lineHeight: 1.2 }}>
                  Nimble 多模态 SOTA 交互体验台
                </Typography>
                <Chip
                  size="small"
                  color="primary"
                  variant="outlined"
                  label="sandwich_v3 · FP32 · Vision-LoRA · 零泄漏锁温"
                  sx={{ height: 22, fontSize: "0.72rem", display: { xs: "none", md: "inline-flex" } }}
                />
              </Stack>
            </Box>

            <Box sx={{ flex: 1 }} />

            <Chip
              size="small"
              variant="outlined"
              label={`测试集 ${sampleCounts.all} ｜ 逆转样本 ${sampleCounts.disagree} ｜ 纯海报 ${sampleCounts.poster_only}`}
              sx={{ display: { xs: "none", lg: "inline-flex" }, fontSize: "0.73rem" }}
            />

            <FormControl size="small" sx={{ minWidth: 255 }}>
              <Select
                value={sampleMode}
                onChange={(e) => {
                  setSampleMode(e.target.value);
                  fetchRandomSample(e.target.value);
                }}
                sx={{ fontSize: "0.8125rem", height: 34 }}
              >
                <MenuItem value="disagree">🎯 经典分歧样本（旧版答错 · SOTA答对）</MenuItem>
                <MenuItem value="poster_only">🖼️ 无标题正文纯海报样本 ({sampleCounts.poster_only})</MenuItem>
                <MenuItem value="all">🎲 全部测试集随机样本 ({sampleCounts.all})</MenuItem>
              </Select>
            </FormControl>

            <Button
              variant="outlined"
              size="small"
              startIcon={<ShuffleRounded />}
              onClick={() => fetchRandomSample()}
              sx={{ height: 34 }}
            >
              换一条样本
            </Button>

            <Tooltip title={themeMode === "light" ? "切换深色模式" : "切换浅色模式"}>
              <IconButton
                size="small"
                onClick={() => setThemeMode((m) => (m === "light" ? "dark" : "light"))}
                sx={{ border: 1, borderColor: "divider", borderRadius: 2 }}
              >
                {themeMode === "light" ? <DarkModeRounded fontSize="small" /> : <LightModeRounded fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>

        {/* Main Two-Column Workspace */}
        <Box
          sx={{
            flex: 1,
            minHeight: 0,
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "45% 55%" },
            overflow: { xs: "auto", md: "hidden" },
          }}
        >
          {/* Left Column: Creative Preview + Modality Ablation + JSON Editors */}
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: 1.5,
              p: 1.5,
              borderRight: { md: 1 },
              borderColor: "divider",
              minHeight: 0,
              overflow: "auto",
            }}
          >
            {/* Ad Creative & Modality Ablation Card */}
            <Card variant="outlined" sx={{ flexShrink: 0 }}>
              <CardContent sx={{ p: 1.5, "&:last-child": { pb: 1.5 } }}>
                <Stack direction="row" spacing={1.75} alignItems="flex-start">
                  {/* Image Thumbnail */}
                  <Box
                    onClick={() => imageUri && setLightboxOpen(true)}
                    sx={{
                      width: 108,
                      height: 108,
                      borderRadius: 2,
                      border: 1,
                      borderColor: "divider",
                      overflow: "hidden",
                      flexShrink: 0,
                      position: "relative",
                      cursor: imageUri ? "zoom-in" : "default",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(255,255,255,0.03)" : "#f8fafc"),
                      "&:hover .zoom-ov": { opacity: 1 },
                    }}
                  >
                    {imageUri ? (
                      <>
                        <Box component="img" src={imageUri} alt="广告原图" sx={{ width: "100%", height: "100%", objectFit: "cover" }} />
                        <Box
                          className="zoom-ov"
                          sx={{
                            position: "absolute",
                            inset: 0,
                            bgcolor: "rgba(0,0,0,0.42)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            opacity: 0,
                            transition: "opacity 0.18s",
                            color: "#fff",
                          }}
                        >
                          <ZoomInRounded />
                        </Box>
                      </>
                    ) : (
                      <Stack alignItems="center" spacing={0.5} sx={{ color: "text.disabled", p: 1 }}>
                        <ImageRounded />
                        <Typography variant="caption" align="center" sx={{ fontSize: "0.68rem" }}>
                          无图片输入
                        </Typography>
                      </Stack>
                    )}
                  </Box>

                  {/* Sample Metadata & Controls */}
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexWrap: "wrap", useFlexGap: true, mb: 1 }}>
                      {currentSample ? (
                        <>
                          <Chip
                            size="small"
                            color="success"
                            icon={<CheckCircleRounded />}
                            label={`真实标签：${currentSample.gold}`}
                            sx={{ fontWeight: 700 }}
                          />
                          <Chip size="small" variant="outlined" label={`ID: ${currentSample.id}`} sx={{ fontFamily: "monospace", fontSize: "0.72rem" }} />
                          {currentSample.is_disagree && (
                            <Chip size="small" color="warning" label="🔥 旧版答错 · SOTA逆转" />
                          )}
                          {!currentSample.has_copy && (
                            <Chip size="small" color="info" label="🖼️ 无文案纯海报" />
                          )}
                        </>
                      ) : (
                        <Chip size="small" color="info" variant="outlined" label={imageUri ? `自定义图片：${imageName}` : "自定义业务输入"} />
                      )}
                    </Stack>

                    {/* Modality Ablation ToggleButtonGroup */}
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.4, fontWeight: 600 }}>
                        模态消融实验切换（一键检验模型真实看图能力）：
                      </Typography>
                      <ToggleButtonGroup
                        size="small"
                        exclusive
                        color="primary"
                        value={modalityMode}
                        onChange={handleModalitySwitch}
                        sx={{
                          flexWrap: "wrap",
                          "& .MuiToggleButton-root": { py: 0.35, px: 1.1, fontSize: "0.75rem", textTransform: "none" },
                        }}
                      >
                        <ToggleButton value="full">
                          <RestoreRounded sx={{ fontSize: 15, mr: 0.5 }} />
                          完整字段
                        </ToggleButton>
                        <ToggleButton value="raw_ad">
                          <ContentCutRounded sx={{ fontSize: 15, mr: 0.5 }} />
                          剥离 OCR/Caption
                        </ToggleButton>
                        <ToggleButton value="pure_image">
                          <VisibilityOffRounded sx={{ fontSize: 15, mr: 0.5 }} />
                          纯看图盲测 (清空文字)
                        </ToggleButton>
                      </ToggleButtonGroup>
                    </Box>

                    {/* Image Upload / Remove / Position */}
                    <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: "wrap", useFlexGap: true }}>
                      <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleUploadImage} />
                      <Button size="small" variant="outlined" startIcon={<UploadFileRounded />} onClick={() => fileInputRef.current?.click()} sx={{ py: 0.25, fontSize: "0.75rem" }}>
                        上传广告图
                      </Button>
                      {imageUri && (
                        <Button
                          size="small"
                          color="error"
                          variant="text"
                          startIcon={<DeleteOutlineRounded />}
                          onClick={() => {
                            setImageUri(null);
                            setImageName("");
                          }}
                          sx={{ py: 0.25, fontSize: "0.75rem" }}
                        >
                          移除图片
                        </Button>
                      )}
                      <Select
                        size="small"
                        value={imgPos}
                        onChange={(e) => setImgPos(e.target.value)}
                        sx={{ height: 27, fontSize: "0.75rem" }}
                      >
                        <MenuItem value="start">image 键位置：开头</MenuItem>
                        <MenuItem value="middle">image 键位置：中间</MenuItem>
                        <MenuItem value="end">image 键位置：结尾</MenuItem>
                      </Select>
                    </Stack>
                  </Box>
                </Stack>
              </CardContent>
            </Card>

            {/* State JSON Editor */}
            <CodeEditorCard
              title="State 业务上下文 (JSON)"
              icon={<DataObjectRounded fontSize="small" color="primary" />}
              value={stateText}
              onChange={(v) => updateStateEditor(v, "custom")}
              error={stateErr}
              minHeight={150}
              actions={
                <Button size="small" variant="text" onClick={formatJsonEditors} sx={{ py: 0, minWidth: 0, fontSize: "0.73rem" }}>
                  格式化
                </Button>
              }
            />

            {/* Questions JSON Editor */}
            <CodeEditorCard
              title="Questions 结构化决策定义 (支持 choice / noul / score)"
              icon={<PsychologyRounded fontSize="small" color="secondary" />}
              value={questionsText}
              onChange={updateQuestionsEditor}
              error={questionsErr}
              minHeight={175}
              actions={
                <Stack direction="row" spacing={0.5}>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<AddCircleOutlineRounded sx={{ fontSize: 14 }} />}
                    onClick={() => addQuestionPreset("noul")}
                    sx={{ py: 0.1, px: 0.8, fontSize: "0.72rem" }}
                  >
                    布尔题 (noul)
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    color="secondary"
                    startIcon={<AddCircleOutlineRounded sx={{ fontSize: 14 }} />}
                    onClick={() => addQuestionPreset("score")}
                    sx={{ py: 0.1, px: 0.8, fontSize: "0.72rem" }}
                  >
                    打分题 (score)
                  </Button>
                </Stack>
              }
            />
          </Box>

          {/* Right Column: Action Toolbar + MUI Tabs + Results */}
          <Box sx={{ display: "flex", flexDirection: "column", minHeight: 0, bgcolor: "background.paper" }}>
            {/* Model & Inference Controls Bar */}
            <Box
              sx={{
                p: 1.5,
                borderBottom: 1,
                borderColor: "divider",
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: 1.25,
                bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(255,255,255,0.015)" : "#fafbfe"),
              }}
            >
              <FormControl size="small" sx={{ minWidth: 270, flex: "1 1 260px" }}>
                <InputLabel id="model-select-label">当前单测模型</InputLabel>
                <Select
                  labelId="model-select-label"
                  label="当前单测模型"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  sx={{ fontSize: "0.82rem" }}
                >
                  {groupedCatalog.map(([groupName, items]) => [
                    <ListSubheader key={`hdr-${groupName}`} sx={{ lineHeight: "28px", fontWeight: 800, fontSize: "0.72rem", color: "primary.main" }}>
                      {groupName}
                    </ListSubheader>,
                    ...items.map((m) => (
                      <MenuItem key={m.id} value={m.id} sx={{ fontSize: "0.8125rem", py: 0.75 }}>
                        <Box sx={{ display: "flex", flexDirection: "column" }}>
                          <Typography variant="body2" sx={{ fontWeight: m.id.includes("sota") ? 700 : 500, fontSize: "0.8125rem" }}>
                            {m.title}
                          </Typography>
                          {m.note && (
                            <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.7rem" }}>
                              {m.note}
                            </Typography>
                          )}
                        </Box>
                      </MenuItem>
                    )),
                  ])}
                </Select>
              </FormControl>

              <Tooltip title="启用在验证集上拟合锁定的零泄漏温度系数 T*，消除微调过自信并校准概率（不改变 argmax 预测）">
                <FormControlLabel
                  control={<Switch size="small" checked={calibrated} onChange={(e) => setCalibrated(e.target.checked)} />}
                  label={<Typography variant="caption" sx={{ fontWeight: 600 }}>零泄漏锁温 (T*)</Typography>}
                  sx={{ mr: 0.5 }}
                />
              </Tooltip>

              <Tooltip title="对 26 个候选选项做 5 次均匀循环移位并融合去均值 Logits，彻底消除位置偏置">
                <FormControlLabel
                  control={<Switch size="small" color="secondary" checked={tta5} onChange={(e) => setTta5(e.target.checked)} />}
                  label={<Typography variant="caption" sx={{ fontWeight: 600 }}>5 视角 TTA5</Typography>}
                  sx={{ mr: 0.5 }}
                />
              </Tooltip>

              <Button
                variant="contained"
                color="primary"
                disabled={isBusy}
                startIcon={<PlayArrowRounded />}
                onClick={() => runSingleModel()}
                sx={{ height: 38, px: 2 }}
              >
                单模型推理 ⌘↵
              </Button>

              <Button
                variant="contained"
                color="secondary"
                disabled={isBusy}
                startIcon={<BoltRounded />}
                onClick={runCompareAll}
                sx={{
                  height: 38,
                  px: 2,
                  background: (t) =>
                    t.palette.mode === "dark"
                      ? "linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)"
                      : "linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%)",
                }}
              >
                一键横评全部 {catalog.length || 10} 种方法
              </Button>
            </Box>

            {isBusy && <LinearProgress color={loadingCompare ? "secondary" : "primary"} sx={{ flexShrink: 0 }} />}

            {/* Navigation Tabs */}
            <Box sx={{ px: 2, borderBottom: 1, borderColor: "divider", display: "flex", alignItems: "center" }}>
              <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ minHeight: 42 }}>
                <Tab label="单模型深度解析" sx={{ minHeight: 42, fontWeight: 600 }} />
                <Tab
                  label={`⚡ ${catalog.length || 10} 方法同题横评`}
                  sx={{ minHeight: 42, fontWeight: 600 }}
                />
                <Tab label="🔍 Prompt & Token 透视" sx={{ minHeight: 42, fontWeight: 600 }} />
              </Tabs>
              <Box sx={{ flex: 1 }} />
              <Tooltip title="复制当前视图原始 JSON">
                <Button size="small" startIcon={<ContentCopyRounded sx={{ fontSize: 15 }} />} onClick={copyCurrentJson}>
                  复制 JSON
                </Button>
              </Tooltip>
            </Box>

            {/* Tab Panels Scroll Area */}
            <Box sx={{ flex: 1, overflow: "auto", p: 2, bgcolor: "background.default" }}>
              {/* TAB 0: Single Model Deep Result */}
              {activeTab === 0 && (
                <>
                  {!singleResult ? (
                    <Paper variant="outlined" sx={{ p: 4, textAlign: "center" }}>
                      <AutoAwesomeRounded color="primary" sx={{ fontSize: 40, mb: 1, opacity: 0.8 }} />
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 0.5 }}>
                        准备就绪：体验单模型推理或一键十模型同题横评
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 520, mx: "auto", mb: 2.5 }}>
                        上方已为您载入测试集样本。您可以点击「一键横评全部方法」在 2 秒内直观对比旧版 v1、纯文本与多模态 SOTA 的预测及置信度差异，或切换左侧「纯看图盲测」观察视觉能力的决定性作用。
                      </Typography>
                      <Stack direction="row" spacing={1.5} justifyContent="center">
                        <Button variant="contained" startIcon={<PlayArrowRounded />} onClick={() => runSingleModel()}>
                          运行当前模型
                        </Button>
                        <Button variant="contained" color="secondary" startIcon={<BoltRounded />} onClick={runCompareAll}>
                          一键横评全部方法
                        </Button>
                      </Stack>
                    </Paper>
                  ) : (
                    <>
                      {/* Gold Comparison Alert */}
                      {singleIsCorrect !== null && (
                        <Alert
                          severity={singleIsCorrect ? "success" : "error"}
                          icon={singleIsCorrect ? <CheckCircleRounded /> : <CancelRounded />}
                          sx={{ mb: 2, fontWeight: 500 }}
                        >
                          <AlertTitle sx={{ fontWeight: 800, mb: 0.25 }}>
                            {singleIsCorrect ? "预测正确（命中测试集真实标签）" : "预测错误（与测试集真实标签不符）"}
                          </AlertTitle>
                          模型选择：<strong>「{singleIndustryAns.choice}」</strong>
                          {!singleIsCorrect && (
                            <>
                              {" "}｜ 测试集真实答案：<strong>「{gold}」</strong>
                            </>
                          )}
                          {singleIndustryAns.probabilities?.[gold] != null && (
                            <>（赋予真实标签概率：<strong>{(singleIndustryAns.probabilities[gold] * 100).toFixed(1)}%</strong>）</>
                          )}
                        </Alert>
                      )}

                      {/* Metadata Chips */}
                      {singleResult.body?.method_meta && (
                        <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: "wrap", useFlexGap: true }}>
                          <Chip size="small" color="primary" label={singleResult.body.method_meta.label || singleResult.body.model} />
                          <Chip size="small" variant="outlined" icon={<CodeRounded />} label={`格式: ${singleResult.body.method_meta.prompt_format}`} />
                          <Chip
                            size="small"
                            variant="outlined"
                            label={singleResult.body.method_meta.multimodal ? "图文多模态" : "纯文本 (忽略图片)"}
                          />
                          <Chip
                            size="small"
                            variant="outlined"
                            label={`候选头: ${singleResult.body.method_meta.fp32_head ? "FP32 精确头" : "BF16 旧头"}`}
                          />
                          <Chip size="small" variant="outlined" label={`温度 T = ${singleResult.body.method_meta.temperature_applied}`} />
                          {singleResult.body.usage && (
                            <Chip size="small" variant="outlined" icon={<TokenRounded />} label={`输入 ${singleResult.body.usage.input_tokens} tok`} />
                          )}
                          {singleResult.body.evaluation_time_ms != null && (
                            <Chip size="small" variant="outlined" icon={<SpeedRounded />} label={`GPU ${singleResult.body.evaluation_time_ms} ms`} />
                          )}
                        </Stack>
                      )}

                      {/* Visual Probability Bars for Every Question */}
                      {singleResult.body?.answers &&
                        Object.entries(singleResult.body.answers).map(([qid, ans]) => (
                          <QuestionDistributionCard
                            key={qid}
                            qid={qid}
                            ans={ans}
                            gold={qid === "industry" ? gold : null}
                          />
                        ))}

                      {/* Raw JSON Accordion */}
                      <Accordion variant="outlined" defaultExpanded={false} sx={{ borderRadius: "10px !important", "&:before": { display: "none" } }}>
                        <AccordionSummary expandIcon={<ExpandMoreRounded />}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                            查看 System One 标准接口响应 JSON (`HTTP {singleResult.status}`)
                          </Typography>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0 }}>
                          <Box
                            component="pre"
                            sx={{
                              m: 0,
                              p: 1.5,
                              borderRadius: 2,
                              overflow: "auto",
                              fontSize: "12px",
                              lineHeight: 1.55,
                              fontFamily: '"JetBrains Mono", ui-monospace, Menlo, monospace',
                              bgcolor: (t) => (t.palette.mode === "dark" ? "#0b0f19" : "#f8fafc"),
                              border: 1,
                              borderColor: "divider",
                            }}
                          >
                            {JSON.stringify(singleResult.body, null, 2)}
                          </Box>
                        </AccordionDetails>
                      </Accordion>
                    </>
                  )}
                </>
              )}

              {/* TAB 1: Compare All 10 Methods */}
              {activeTab === 1 && (
                <>
                  {!compareResult ? (
                    <Paper variant="outlined" sx={{ p: 4, textAlign: "center" }}>
                      <BoltRounded color="secondary" sx={{ fontSize: 40, mb: 1 }} />
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
                        {loadingCompare ? "正在同一条样本上并行评测全部模型（约 1~2 秒）…" : "尚未运行全方法横评"}
                      </Typography>
                      {!loadingCompare && (
                        <Button variant="contained" color="secondary" startIcon={<BoltRounded />} onClick={runCompareAll}>
                          立即横评全部方法
                        </Button>
                      )}
                    </Paper>
                  ) : (
                    <>
                      {/* 4 Summary KPI Cards */}
                      {compareKpis && (
                        <Box
                          sx={{
                            display: "grid",
                            gridTemplateColumns: { xs: "1fr 1fr", lg: "repeat(4, 1fr)" },
                            gap: 1.25,
                            mb: 2,
                          }}
                        >
                          <Paper variant="outlined" sx={{ p: 1.5 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                              本道题命中模型数
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 800, mt: 0.25 }}>
                              {compareKpis.correctCount != null ? `${compareKpis.correctCount} / ${compareKpis.total}` : `${compareKpis.total} 个模型`}
                            </Typography>
                            <Typography variant="caption" color="success.main">
                              {compareKpis.gold ? `真实答案：${compareKpis.gold}` : "自定义无标签样本"}
                            </Typography>
                          </Paper>

                          <Paper variant="outlined" sx={{ p: 1.5 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                              4B 多模态 vs 纯文本 (真实类概率)
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 800, mt: 0.25, color: "primary.main" }}>
                              {compareKpis.pGoldSota4b != null && compareKpis.pGoldText4b != null
                                ? `${(compareKpis.pGoldSota4b * 100).toFixed(1)}% vs ${(compareKpis.pGoldText4b * 100).toFixed(1)}%`
                                : "—"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {compareKpis.pGoldSota4b != null && compareKpis.pGoldText4b != null
                                ? `视觉信号增益 ${((compareKpis.pGoldSota4b - compareKpis.pGoldText4b) * 100 >= 0 ? "+" : "") + ((compareKpis.pGoldSota4b - compareKpis.pGoldText4b) * 100).toFixed(1)} pp`
                                : "对比多模态与纯文本置信度"}
                            </Typography>
                          </Paper>

                          <Paper variant="outlined" sx={{ p: 1.5 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                              sandwich_v3 vs 旧版 v1 长度
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 800, mt: 0.25 }}>
                              {compareKpis.tokV3 && compareKpis.tokV1 ? `${compareKpis.tokV3} vs ${compareKpis.tokV1}` : "—"}
                            </Typography>
                            <Typography variant="caption" color="success.main">
                              {compareKpis.tokV3 && compareKpis.tokV1
                                ? `消除双重 JSON 节省 ${compareKpis.tokV1 - compareKpis.tokV3} tok (-${(((compareKpis.tokV1 - compareKpis.tokV3) / compareKpis.tokV1) * 100).toFixed(1)}%)`
                                : "Token 紧凑度对比"}
                            </Typography>
                          </Paper>

                          <Paper variant="outlined" sx={{ p: 1.5 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                              全量横评总耗时 & 配置
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 800, mt: 0.25 }}>
                              {compareResult.total_wall_ms} ms
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              锁温校准: {compareResult.calibrated ? "开启" : "关闭"} · 视角: {compareResult.tta_views}x
                            </Typography>
                          </Paper>
                        </Box>
                      )}

                      {/* Comparison MUI Table */}
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableHead>
                            <TableRow sx={{ bgcolor: (t) => (t.palette.mode === "dark" ? "rgba(255,255,255,0.03)" : "#f8fafc") }}>
                              <TableCell>方法 / 模型</TableCell>
                              <TableCell>协议 & 温度</TableCell>
                              <TableCell>预测结论</TableCell>
                              <TableCell sx={{ minWidth: 120 }}>Top-1 概率</TableCell>
                              <TableCell sx={{ minWidth: 120 }}>真实标签概率</TableCell>
                              <TableCell align="right">Token</TableCell>
                              <TableCell align="right">延迟</TableCell>
                              <TableCell align="center">透视</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {compareResult.results.map((item) => {
                              const catInfo = catalog.find((c) => c.id === item.model) || {};
                              const isSota = item.model.includes("sota") && !item.model.includes("text");
                              if (!item.ok) {
                                return (
                                  <TableRow key={item.model}>
                                    <TableCell>{catInfo.title || item.model}</TableCell>
                                    <TableCell colSpan={7} sx={{ color: "error.main" }}>错误: {item.error}</TableCell>
                                  </TableRow>
                                );
                              }
                              const qid = compareKpis?.qid || "industry";
                              const res = item.result;
                              const ans = res.answers?.[qid] || {};
                              const meta = res.method_meta || {};
                              const pred =
                                ans.type === "choice"
                                  ? ans.choice
                                  : ans.type === "noul"
                                  ? `P(true)=${ans.noul}`
                                  : `score=${ans.score}`;
                              const probs = ans.probabilities || {};
                              const topP = ans.type === "choice" && probs[ans.choice] != null ? probs[ans.choice] : null;
                              const goldP = gold && probs[gold] != null ? probs[gold] : null;
                              const isOk = gold && ans.type === "choice" ? pred === gold : null;

                              return (
                                <TableRow
                                  key={item.model}
                                  hover
                                  sx={{
                                    bgcolor: isSota
                                      ? (t) => alpha(t.palette.primary.main, t.palette.mode === "dark" ? 0.08 : 0.035)
                                      : "inherit",
                                  }}
                                >
                                  <TableCell>
                                    <Typography variant="body2" sx={{ fontWeight: isSota ? 700 : 600, fontSize: "0.8rem" }}>
                                      {catInfo.title || item.model}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.7rem" }}>
                                      {catInfo.note || item.model}
                                    </Typography>
                                  </TableCell>
                                  <TableCell>
                                    <Stack spacing={0.25}>
                                      <Typography variant="caption" sx={{ fontFamily: "monospace", fontWeight: 600 }}>
                                        {meta.prompt_format}
                                      </Typography>
                                      <Typography variant="caption" color="text.secondary">
                                        {meta.multimodal ? "图文" : "纯文本"} · T={meta.temperature_applied}
                                      </Typography>
                                    </Stack>
                                  </TableCell>
                                  <TableCell>
                                    {isOk !== null ? (
                                      <Chip
                                        size="small"
                                        color={isOk ? "success" : "error"}
                                        variant={isOk ? "filled" : "outlined"}
                                        label={`${isOk ? "✓" : "✗"} ${pred}`}
                                        sx={{ fontWeight: 700, maxWidth: 175 }}
                                      />
                                    ) : (
                                      <Chip size="small" variant="outlined" label={pred} />
                                    )}
                                  </TableCell>
                                  <TableCell>
                                    {topP != null ? (
                                      <Box>
                                        <Typography variant="caption" sx={{ fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
                                          {(topP * 100).toFixed(1)}%
                                        </Typography>
                                        <LinearProgress
                                          variant="determinate"
                                          value={Math.max(1, topP * 100)}
                                          color={isOk === false ? "error" : "primary"}
                                          sx={{ height: 5, borderRadius: 99, mt: 0.3 }}
                                        />
                                      </Box>
                                    ) : (
                                      "—"
                                    )}
                                  </TableCell>
                                  <TableCell>
                                    {goldP != null ? (
                                      <Box>
                                        <Typography variant="caption" sx={{ fontWeight: 700, color: "success.main", fontVariantNumeric: "tabular-nums" }}>
                                          {(goldP * 100).toFixed(1)}%
                                        </Typography>
                                        <LinearProgress
                                          variant="determinate"
                                          value={Math.max(1, goldP * 100)}
                                          color="success"
                                          sx={{ height: 5, borderRadius: 99, mt: 0.3 }}
                                        />
                                      </Box>
                                    ) : (
                                      "—"
                                    )}
                                  </TableCell>
                                  <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums", fontFamily: "monospace" }}>
                                    {res.usage?.input_tokens}
                                  </TableCell>
                                  <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>
                                    {res.evaluation_time_ms} ms
                                  </TableCell>
                                  <TableCell align="center" sx={{ whiteSpace: "nowrap" }}>
                                    <Button
                                      size="small"
                                      variant="outlined"
                                      onClick={() => fetchPromptOnly(item.model)}
                                      sx={{ py: 0.2, px: 1, fontSize: "0.72rem", minWidth: 0 }}
                                    >
                                      Prompt
                                    </Button>
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

              {/* TAB 2: Prompt & Token Stream Inspector */}
              {activeTab === 2 && (
                <>
                  {/* Quick In-Tab Model Switcher for Instant Prompt Format Diffing */}
                  <Paper variant="outlined" sx={{ p: 1.5, mb: 2, display: "flex", alignItems: "center", gap: 1.5, flexWrap: "wrap" }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                      切换模型实时对比底层 Prompt 构造：
                    </Typography>
                    <FormControl size="small" sx={{ minWidth: 280 }}>
                      <Select
                        value={promptResult?.model || selectedModel}
                        onChange={(e) => fetchPromptOnly(e.target.value)}
                        sx={{ fontSize: "0.8rem" }}
                      >
                        {catalog.map((m) => (
                          <MenuItem key={m.id} value={m.id} sx={{ fontSize: "0.8rem" }}>
                            {m.title}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Button size="small" variant="outlined" onClick={() => fetchPromptOnly(selectedModel)}>
                      刷新当前 Prompt
                    </Button>
                  </Paper>

                  {!promptResult ? (
                    <Paper variant="outlined" sx={{ p: 4, textAlign: "center" }}>
                      <CodeRounded color="primary" sx={{ fontSize: 40, mb: 1 }} />
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
                        点击下方按钮透视当前样本的底层 Token 序列与视觉 Patch 网格
                      </Typography>
                      <Button variant="contained" onClick={() => fetchPromptOnly(selectedModel)}>
                        生成 Prompt 透视图
                      </Button>
                    </Paper>
                  ) : (
                    <>
                      {/* Summary Chips */}
                      <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: "wrap", useFlexGap: true }}>
                        <Chip size="small" color="primary" label={`模型: ${promptResult.model}`} />
                        <Chip size="small" color="secondary" variant="outlined" label={`协议: ${promptResult.summary?.prompt_format}`} />
                        <Chip size="small" variant="outlined" label={`总 Token: ${promptResult.summary?.total_tokens}`} />
                        <Chip size="small" color="info" variant="outlined" label={`视觉 <|image_pad|>: ${promptResult.summary?.image_tokens}`} />
                        {promptResult.rendered?.image?.grid_patches_thw && (
                          <Chip
                            size="small"
                            variant="outlined"
                            label={`视觉网格 T×H×W: ${promptResult.rendered.image.grid_patches_thw.join("×")}`}
                          />
                        )}
                        <Chip
                          size="small"
                          variant="outlined"
                          label={`FP32 候选头: ${promptResult.summary?.fp32_head ? "是" : "否 (BF16)"}`}
                        />
                      </Stack>

                      {/* Structured Segment Stream */}
                      <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.25 }}>
                          1. 结构化分段 Token 流（紫色为特殊控制符 · 蓝色为多模态视觉 Patch · 灰底为文本内容）
                        </Typography>
                        {(() => {
                          const groupedSegs = [];
                          for (const g of promptResult.segments || []) {
                            if (!groupedSegs.length || groupedSegs[groupedSegs.length - 1].name !== g.segment) {
                              groupedSegs.push({ name: g.segment, items: [] });
                            }
                            groupedSegs[groupedSegs.length - 1].items.push(g);
                          }
                          return groupedSegs.map((sg, idx) => (
                            <Box key={idx} sx={{ mb: 2 }}>
                              <Divider textAlign="left" sx={{ mb: 1 }}>
                                <Chip size="small" label={sg.name} sx={{ fontFamily: "monospace", fontSize: "0.72rem" }} />
                              </Divider>
                              <Box
                                sx={{
                                  p: 1.5,
                                  borderRadius: 2,
                                  bgcolor: (t) => (t.palette.mode === "dark" ? "#0b0f19" : "#f8fafc"),
                                  border: 1,
                                  borderColor: "divider",
                                  fontFamily: '"JetBrains Mono", ui-monospace, Menlo, monospace',
                                  fontSize: "12px",
                                  lineHeight: 1.75,
                                  whiteSpace: "pre-wrap",
                                  wordBreak: "break-word",
                                }}
                              >
                                {sg.items.map((it, k) => {
                                  if (it.kind === "special") {
                                    return (
                                      <Chip
                                        key={k}
                                        size="small"
                                        color="secondary"
                                        variant="outlined"
                                        label={it.text}
                                        sx={{ mx: 0.25, my: 0.2, height: 20, fontFamily: "monospace", fontSize: "11px" }}
                                      />
                                    );
                                  }
                                  if (it.kind === "image") {
                                    return (
                                      <Chip
                                        key={k}
                                        size="small"
                                        color="info"
                                        icon={<ImageRounded sx={{ fontSize: 14 }} />}
                                        label={`${it.text} × ${it.count} 视觉 Tokens`}
                                        sx={{ mx: 0.4, my: 0.25, height: 22, fontFamily: "monospace", fontSize: "11.5px", fontWeight: 700 }}
                                      />
                                    );
                                  }
                                  return <span key={k}>{it.text}</span>;
                                })}
                              </Box>
                            </Box>
                          ));
                        })()}
                      </Paper>

                      {/* Token Table Accordion */}
                      <Accordion variant="outlined" sx={{ borderRadius: "10px !important", "&:before": { display: "none" } }}>
                        <AccordionSummary expandIcon={<ExpandMoreRounded />}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                            2. 展开逐 Token 词表映射明细（共 {promptResult.tokens?.length || 0} 个 Token）
                          </Typography>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0 }}>
                          <TableContainer sx={{ maxHeight: 420 }}>
                            <Table size="small" stickyHeader>
                              <TableHead>
                                <TableRow>
                                  <TableCell>#</TableCell>
                                  <TableCell>Token ID</TableCell>
                                  <TableCell>Token Piece</TableCell>
                                  <TableCell>所属段落</TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {(showAllTokens ? promptResult.tokens : (promptResult.tokens || []).slice(0, 250)).map((tok) => (
                                  <TableRow key={tok.i} hover>
                                    <TableCell sx={{ fontFamily: "monospace", color: "text.secondary" }}>{tok.i}</TableCell>
                                    <TableCell sx={{ fontFamily: "monospace" }}>{tok.id}</TableCell>
                                    <TableCell sx={{ fontFamily: "monospace", fontWeight: 600 }}>{tok.piece}</TableCell>
                                    <TableCell sx={{ color: "text.secondary" }}>{tok.segment}</TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                          {(promptResult.tokens?.length || 0) > 250 && !showAllTokens && (
                            <Box sx={{ textAlign: "center", pt: 1.5 }}>
                              <Button size="small" variant="outlined" onClick={() => setShowAllTokens(true)}>
                                展开剩余 {promptResult.tokens.length - 250} 个 Token
                              </Button>
                            </Box>
                          )}
                        </AccordionDetails>
                      </Accordion>
                    </>
                  )}
                </>
              )}
            </Box>
          </Box>
        </Box>

        {/* Image Lightbox Dialog */}
        <Dialog open={lightboxOpen} onClose={() => setLightboxOpen(false)} maxWidth="lg">
          <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1.25 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              广告素材高清预览 {currentSample ? `(${currentSample.id} · 真实标签：${currentSample.gold})` : ""}
            </Typography>
            <IconButton size="small" onClick={() => setLightboxOpen(false)}>
              <CloseRounded />
            </IconButton>
          </DialogTitle>
          <DialogContent dividers sx={{ textAlign: "center", bgcolor: "#0b0f19", p: 2 }}>
            {imageUri && (
              <Box
                component="img"
                src={imageUri}
                alt="广告大图"
                sx={{ maxWidth: "86vw", maxHeight: "80vh", objectFit: "contain", borderRadius: 1 }}
              />
            )}
          </DialogContent>
        </Dialog>

        {/* Toast Notification */}
        <Snackbar
          open={Boolean(toast)}
          autoHideDuration={3200}
          onClose={() => setToast("")}
          message={toast}
          anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        />
      </Box>
    </ThemeProvider>
  );
}
