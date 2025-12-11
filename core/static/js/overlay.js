// Company ticker lookup map
let companyTickerMap = {};

// Load company ticker mappings (accepts either {ticker: title} or [{ticker,title}])
async function loadCompanyTickers() {
  try {
    const res = await fetch('/company_tickers.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const mapping = {};

    if (Array.isArray(data)) {
      data.forEach((item) => {
        const ticker = (item && (item.ticker || item.symbol)) || '';
        const title = item && (item.title || item.name);
        if (ticker && title) {
          mapping[String(ticker).toUpperCase()] = String(title).trim();
        }
      });
    } else if (data && typeof data === 'object') {
      Object.entries(data).forEach(([key, value]) => {
        if (value && typeof value === 'string') {
          mapping[String(key).toUpperCase()] = value;
        } else if (value && typeof value === 'object') {
          const ticker = value.ticker || value.symbol || key;
          const title = value.title || value.name;
          if (ticker && title) {
            mapping[String(ticker).toUpperCase()] = String(title).trim();
          }
        }
      });
    }

    companyTickerMap = mapping;
  } catch (err) {
    console.warn('Could not load company tickers:', err);
  }
}

loadCompanyTickers();

// Helpers available to refresh()
function normalizeTxn(value) {
  if (!value) return "";
  const v = String(value).toLowerCase();
  if (v.includes("buy") || v.includes("purchase")) return "Buy";
  if (v.includes("sell") || v.includes("sale")) return "Sell";
  if (v.includes("partial")) return "Sell"; // treat partial sales as Sell
  return value; // fallback to original text
}

function preferNetWorth(data) {
  const s = data.normalized_net_worth || data.networth_current || "N/A";
  return normalizeMoney(s);
}
const DEGREE_KEYWORD_REGEX = /\b(AA|A\.A\.|AAS|A\.A\.S\.|AS|A\.S\.|AB|A\.B\.|BA|B\.A\.|Bachelor|BS|B\.S\.|BSc|B\.Sc\.|BBA|B\.B\.A\.|BFA|B\.F\.A\.|MA|M\.A\.|MS|M\.S\.|MSc|M\.Sc\.|MBA|M\.B\.A\.|JD|J\.D\.|LLB|LL\.B\.|LLM|LL\.M\.|MD|M\.D\.|DO|D\.O\.|DDS|D\.D\.S\.|DMD|D\.M\.D\.|DVM|D\.V\.M\.|PhD|Ph\.D\.|EdD|Ed\.D\.|DPhil|D\.Phil\.|ScD|Sc\.D\.|DrPH|Dr\.P\.H\.|MFA|M\.F\.A\.|MSW|M\.S\.W\.|MPA|M\.P\.A\.|MPP|M\.P\.P\.|MEd|M\.Ed\.|EdM|Ed\.M\.|MEng|M\.Eng\.|BEng|B\.Eng\.|BSN|B\.S\.N\.|MSN|M\.S\.N\.|DNP|D\.N\.P\.|Doctor of|Master of|Bachelor of|Juris Doctor|Associate of|Associate's|Associates)\b/i;
const HIGH_SCHOOL_REGEX = /(High School|Highschool|Secondary School|Prep School|Preparatory School|Middle School|Elementary School)/i;
const ACADEMY_ALLOWLIST_REGEX = /\b(United States|U\.S\.|US|Naval|Air Force|Military|Coast Guard|Merchant Marine|Service|Fine Arts|Fine Art|Art Institute|Art College|Art University|Academy of Art|Academy of Fine Arts|Academy of Arts|Academy of Music|Academy of Science|Academy of Sciences)\b/i;

function isHighSchoolLine(text) {
  if (!text) return false;
  const normalized = text.trim();
  if (!normalized) return false;
  if (HIGH_SCHOOL_REGEX.test(normalized)) return true;
  if (/\bAcademy\b/i.test(normalized) && !ACADEMY_ALLOWLIST_REGEX.test(normalized)) {
    return true;
  }
  return false;
}

function isDegreeLine(text) {
  if (!text) return false;
  const normalized = text.trim();
  if (!normalized || isHighSchoolLine(normalized)) return false;
  if (/^residency\b/i.test(normalized)) return true;
  if (/^fellowship\b/i.test(normalized)) return true;
  return DEGREE_KEYWORD_REGEX.test(normalized);
}

function splitDegreeText(value) {
  if (!value) return [];
  return String(value)
    .split(/(?:\r?\n|[\u2022\u00B7;])/)
    .map(part => part.trim())
    .filter(Boolean);
}

function normalizeDegreeLabel(raw) {
  if (!raw) return '';
  const text = String(raw).trim();
  const rules = [
    [/Bachelor(?:'s)? of Science/i, 'BS'],
    [/Bachelor(?:'s)? of Arts/i, 'BA'],
    [/Bachelor(?:'s)? of Business Administration/i, 'BBA'],
    [/Bachelor(?:'s)? of Fine Arts/i, 'BFA'],
    [/Bachelor(?:'s)? of Engineering/i, 'BEng'],
    [/Bachelor(?:'s)? of Laws/i, 'LLB'],
    [/Master(?:'s)? of Science/i, 'MS'],
    [/Master(?:'s)? of Arts/i, 'MA'],
    [/Master of Business Administration/i, 'MBA'],
    [/Master of Public Administration/i, 'MPA'],
    [/Master of Public Policy/i, 'MPP'],
    [/Master of Fine Arts/i, 'MFA'],
    [/Master of Engineering/i, 'MEng'],
    [/Doctor of Medicine/i, 'MD'],
    [/Doctor of Osteopathic Medicine/i, 'DO'],
    [/Doctor of Philosophy/i, 'PhD'],
    [/Doctor of Education/i, 'EdD'],
    [/Doctor of Dental Surgery/i, 'DDS'],
    [/Doctor of Veterinary Medicine/i, 'DVM'],
    [/Doctor of Pharmacy/i, 'PharmD'],
    [/Doctor of Law/i, 'JD'],
    [/Juris Doctor/i, 'JD'],
    [/Doctor of Science/i, 'ScD'],
    [/Associate(?:'s)? of Arts/i, 'AA'],
    [/Associate(?:'s)? of Science/i, 'AS'],
  ];
  for (const [pattern, value] of rules) {
    if (pattern.test(text)) return value;
  }
  // If already an abbreviation like B.S. or M.D.
  const compact = text.replace(/\./g, '').replace(/\s+/g, '');
  if (compact.length <= 5 && /^[A-Za-z]+$/.test(compact)) {
    return compact.toUpperCase();
  }
  return text.replace(/\s{2,}/g, ' ').trim();
}

function simplifyInstitutionName(name) {
  let s = (name || '').trim();
  if (!s) return '';

  s = s.replace(/(University)\s+Law Center\b/gi, '$1');
  s = s.replace(/(University)\s+School of Foreign Service\b/gi, '$1');
  s = s.replace(/,\s*School of Foreign Service\b/gi, '');
  s = s.replace(/\bSchool of Foreign Service\b/gi, '');
  s = s.replace(/,\s*Law Center\b/gi, '');
  s = s.replace(/\bLaw Center\b/gi, '');
  s = s.replace(/,\s*Institute of Politics\b/gi, '');
  s = s.replace(/\bInstitute of Politics\b/gi, '');

  s = s.replace(/\s{2,}/g, ' ').trim();
  s = s.replace(/,\s*,/g, ', ');
  s = s.replace(/,\s*$/g, '');
  s = s.replace(/\bUniversity\s+University\b/gi, 'University');

  s = s.trim();
  if (!s) {
    return (name || '').trim();
  }
  return s;
}

function formatDegreeEntry(raw) {
  const original = toText(raw);
  if (!original) return '';

  let s = original.replace(/^[\-\u2013\u2014\u2022•]+\s*/, '').trim();
  if (!s) return '';

  let descriptor = '';
  if (/^residency\b/i.test(s)) {
    descriptor = 'Residency';
    s = s.replace(/^residency\s+(?:at|in)\s*/i, '');
  } else if (/^fellowship\b/i.test(s)) {
    descriptor = 'Fellowship';
    s = s.replace(/^fellowship\s+(?:at|in)\s*/i, '');
  }

  s = s.replace(/^(?:graduated|graduate of|earned|received|completed|studied|enrolled)\s+/i, '');
  s = s.replace(/^(?:with\s+)?(?:a|an)\s+/i, '');

  let honors = '';
  const honorsMatch = s.match(/\b(?:summa|magna)?\s*cum laude\b/i);
  if (honorsMatch) {
    honors = honorsMatch[0].replace(/\s+/g, ' ').trim();
    s = s.replace(honorsMatch[0], '').trim();
  }

  let year = '';
  const yearMatch = s.match(/\b(19|20)\d{2}\b/);
  if (yearMatch) {
    year = yearMatch[0];
    s = s.replace(yearMatch[0], '').trim();
  }

  let degree = '';
  const degreeMatch = original.match(DEGREE_KEYWORD_REGEX);
  if (degreeMatch) {
    degree = normalizeDegreeLabel(degreeMatch[0]);
    const escaped = degreeMatch[0].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    s = s.replace(new RegExp(escaped, 'i'), '').trim();
  }

  s = s.replace(/\bfrom\b\s+/gi, '');
  s = s.replace(/\bat\b\s+/gi, '');
  s = s.replace(/\s{2,}/g, ' ').trim();
  s = s.replace(/[,;]\s*$/g, '');

  const segments = s.split(',').map(part => part.trim()).filter(Boolean);
  let institution = '';
  if (segments.length) {
    institution = segments.shift() || '';
    const extra = segments.filter(seg => /\b(University|College|School|Institute|Academy|Faculty|Center|Centre)\b/i.test(seg));
    if (extra.length) {
      institution += ', ' + extra.join(', ');
    }
  } else {
    institution = s;
  }

  institution = simplifyInstitutionName(institution);

  const parts = [];
  if (descriptor && degree) {
    parts.push(`${descriptor}, ${degree}`);
  } else if (descriptor) {
    parts.push(descriptor);
  } else if (degree) {
    parts.push(degree);
  }
  if (!parts.length && degree) {
    parts.push(degree);
  }
  if (institution) {
    parts.push(institution);
  }
  if (honors) {
    parts.push(honors);
  }
  if (year) {
    parts.push(year);
  }

  const result = parts.join(', ').replace(/\s{2,}/g, ' ').trim();
  return result || original;
}

function collectEducationEntries(items, startIndex) {
  const degrees = [];
  const seen = new Set();

  const addDegree = (raw) => {
    if (!raw) return;
    const formatted = formatDegreeEntry(raw);
    if (!formatted) return;
    const key = formatted.toLowerCase();
    if (seen.has(key)) return;
    if (!isDegreeLine(raw)) return;
    seen.add(key);
    degrees.push(formatted);
  };

  const startItem = items[startIndex];
  if (startItem && startItem.type === 'pair' && startItem.value) {
    splitDegreeText(startItem.value).forEach(addDegree);
  }

  let index = startIndex + 1;
  while (index < items.length) {
    const next = items[index];
    if (!next) break;
    if (next.type === 'text') {
      const raw = toText(next.text);
      if (!raw) {
        index += 1;
        continue;
      }
      if (isHighSchoolLine(raw)) {
        index += 1;
        continue;
      }
      if (isDegreeLine(raw)) {
        addDegree(raw);
        index += 1;
        continue;
      }
      break;
    }
    if (next.type === 'pair') {
      break;
    }
    index += 1;
  }

  return { degrees, nextIndex: index };
}
function fmtDate(s) {
  const raw = (s || "").trim();
  if (!raw) return "";
  
  const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  
  // Try ISO/standard parse first
  const d = new Date(raw);
  if (!isNaN(d)) {
    const m = d.getUTCMonth();
    const y = d.getUTCFullYear();
    const day = d.getUTCDate();
    if (y && y > 1900) {
      const result = { month: MONTHS_SHORT[m], year: String(y) };
      if (!Number.isNaN(day) && day) {
        result.day = String(day);
      }
      return result;
    }
  }
  
  // Fallback: extract YYYY-MM or YYYY/MM
  const m1 = raw.match(/^(\d{4})[-\/.](\d{1,2})/);
  if (m1) {
    const y = parseInt(m1[1], 10);
    const m = parseInt(m1[2], 10);
    if (y && m && m >= 1 && m <= 12) {
      return { month: MONTHS_SHORT[m - 1], year: String(y) };
    }
  }
  
  // Return raw as fallback
  return raw;
}

// Month Day, Year (e.g., January 2, 2024) format for birth dates
function fmtDateLongMDY(s) {
  const raw = (s || "").trim();
  if (!raw) return "";
  const MONTHS = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
  ];
  // Try native parse first
  let d = new Date(raw);
  if (!isNaN(d)) {
    const y = d.getUTCFullYear();
    const m = d.getUTCMonth();
    const day = d.getUTCDate();
    if (y && y > 0 && m >= 0 && day > 0) {
      return `${MONTHS[m]} ${day}, ${y}`;
    }
  }
  // Try YYYY-MM-DD or YYYY/MM/DD
  let m = raw.match(/^(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})$/);
  if (m) {
    const y = parseInt(m[1], 10);
    const mo = Math.max(1, Math.min(12, parseInt(m[2], 10))) - 1;
    const day = Math.max(1, Math.min(31, parseInt(m[3], 10)));
    return `${MONTHS[mo]} ${day}, ${y}`;
  }
  // Try MM/DD/YYYY
  m = raw.match(/^(\d{1,2})[\/.](\d{1,2})[\/.](\d{4})$/);
  if (m) {
    const mo = Math.max(1, Math.min(12, parseInt(m[1], 10))) - 1;
    const day = Math.max(1, Math.min(31, parseInt(m[2], 10)));
    const y = parseInt(m[3], 10);
    return `${MONTHS[mo]} ${day}, ${y}`;
  }
  return raw; // fallback: display as-is
}

function pluralize(n, singular) {
  return `${n} ${singular}${n === 1 ? '' : 's'}`;
}

// Normalize tenure strings to "X years, Y months" with proper pluralization
function formatTenurePretty(raw) {
  if (!raw) return '';
  const s = String(raw).trim();
  const lower = s.toLowerCase();
  // Common patterns: "12 years, 3 months", "12 yrs 3 mos", "12y 3m", etc.
  const yMatch = lower.match(/(\d+)\s*(?:years?|yrs?|y)\b/);
  const mMatch = lower.match(/(\d+)\s*(?:months?|mos?|m)\b/);
  let years = yMatch ? parseInt(yMatch[1], 10) : null;
  let months = mMatch ? parseInt(mMatch[1], 10) : null;
  // Fallback: if no explicit units, try two numbers separated by space/comma
  if (years === null && months === null) {
    const nums = lower.match(/(\d{1,2})[^\d]+(\d{1,2})/);
    if (nums) {
      years = parseInt(nums[1], 10);
      months = parseInt(nums[2], 10);
    } else {
      const single = lower.match(/(\d{1,2})/);
      if (single) years = parseInt(single[1], 10);
    }
  }
  const out = [];
  if (typeof years === 'number') out.push(pluralize(years, 'year'));
  if (typeof months === 'number') out.push(pluralize(months, 'month'));
  return out.join(', ') || s;
}

function toText(value) {
  return (value === undefined || value === null) ? '' : String(value).trim();
}

const PLACEHOLDER = '\u2014';

function textOrPlaceholder(value) {
  const text = toText(value);
  return text ? text : PLACEHOLDER;
}

function moneyOrPlaceholder(value) {
  const normalized = normalizeMoney(value || '');
  if (!normalized) {
    return PLACEHOLDER;
  }
  return normalized;
}

function setSectionState(sectionEl, hasContent) {
  if (!sectionEl) {
    return;
  }
  sectionEl.style.display = hasContent ? 'block' : 'none';
  sectionEl.dataset.hasContent = hasContent ? 'true' : 'false';
}

function ensureKvList(listEl) {
  if (listEl) {
    listEl.classList.add('list--kv');
  }
}

const TRADE_RANGE_BY_LOWER = new Map([
  [0, "$0 - $1,000"],
  [1, "$0 - $1,000"],
  [1001, "$1,001 - $15,000"],
  [15001, "$15,001 - $50,000"],
  [50001, "$50,001 - $100,000"],
  [100001, "$100,001 - $250,000"],
  [250001, "$250,001 - $500,000"],
  [500001, "$500,001 - $1,000,000"],
  [1000001, "$1,000,001 - $5,000,000"],
  [5000001, "$5,000,001 - $25,000,000"],
  [25000001, "$25,000,001 - $50,000,000"],
  [50000001, "$50,000,001+"],
]);

const TICKER_SCROLL_STEP = 1;
const TICKER_SCROLL_INTERVAL_MS = 30;
const TICKER_SCROLL_START_DELAY_MS = 5000;
const TICKER_SCROLL_BOTTOM_PAUSE_MS = 5000;

function normalizeMoney(value) {
  if (value == null) return "";
  let raw = String(value).trim();
  if (!raw) return "";

  const lowered = raw.toLowerCase();
  if (lowered === "n/a" || lowered === "na" || lowered === "unknown") {
    return "N/A";
  }

  const parts = raw.split(/[–—-]/).map(p => p.trim()).filter(Boolean);
  if (raw.match(/[–—-]/) && parts.length === 2) {
    const [lo, hi] = parts;
    const loFmt = normalizeMoney(lo);
    const hiFmt = normalizeMoney(hi);
    if (loFmt && hiFmt) {
      return `${loFmt} - ${hiFmt}`;
    }
  }

  let s = raw;
  let negative = false;
  if (s.startsWith('-')) {
    negative = true;
    s = s.slice(1).trim();
  }

  s = s.replace(/^\$+/, '').replace(/,/g, '');
  if (!s) {
    return negative ? '-$0' : '$0';
  }

  const num = Number(s);
  if (!Number.isFinite(num)) {
    return (negative ? '-' : '') + raw.replace(/^\$+/, '$');
  }

  const isInt = Math.abs(num - Math.trunc(num)) < 1e-9;
  const formatted = Math.abs(num).toLocaleString('en-US', {
    minimumFractionDigits: isInt ? 0 : 2,
    maximumFractionDigits: isInt ? 0 : 2,
  });

  return `${negative ? '-' : ''}$${formatted}`;
}

// -------------------------
// Ticker mode helpers
// -------------------------
const tickerState = {
  active: false,
  sectionIndex: 0,
  scrollTimer: null,
  scrollDelayTimer: null,
  scrollPauseTimer: null,
  dataHash: "",
  sections: [],
};

function stopTickerScroll() {
  if (tickerState.scrollTimer) {
    clearInterval(tickerState.scrollTimer);
    tickerState.scrollTimer = null;
  }
  if (tickerState.scrollDelayTimer) {
    clearTimeout(tickerState.scrollDelayTimer);
    tickerState.scrollDelayTimer = null;
  }
  if (tickerState.scrollPauseTimer) {
    clearTimeout(tickerState.scrollPauseTimer);
    tickerState.scrollPauseTimer = null;
  }
}

function stopTickerTimers() {
  stopTickerScroll();
  tickerState.sections = [];
}

function exitTickerMode() {
  stopTickerTimers();
  tickerState.active = false;
  const ticker = document.getElementById('ticker-mode');
  const grid = document.getElementById('card-container');
  const networthSection = document.getElementById('networth-section');
  const header = document.querySelector('.overlay-header');
  const overlayEl = document.getElementById('overlay');
  if (ticker) ticker.style.display = 'none';
  if (grid) grid.style.display = '';
  if (networthSection) networthSection.style.display = '';
  if (header) header.style.display = '';
  if (overlayEl) overlayEl.classList.remove('overlay--ticker-mode');
  tickerState.sectionIndex = 0;
  tickerState.dataHash = "";
  tickerState.sections = [];
  reconcileCardRotation();
}

function prepareTickerLoop(listEl) {
  if (!listEl) return 0;
  const wrapper = listEl.parentElement;
  if (!wrapper) return 0;

  const baseHeight = listEl.scrollHeight;
  listEl.dataset.baseHeight = String(baseHeight || 0);

  return baseHeight;
}

function startTickerScroll(listEl) {
  if (!listEl) return;
  const wrapper = listEl.parentElement;
  if (!wrapper) return;
  stopTickerScroll();
  wrapper.scrollTop = 0;
  const scrollHeight = listEl.scrollHeight;
  const canScroll = scrollHeight > (wrapper.clientHeight + 1);
  if (!canScroll) {
    if ((tickerState.sections || []).length > 1) {
      tickerState.scrollPauseTimer = setTimeout(() => {
        tickerState.scrollPauseTimer = null;
        advanceTickerSection();
      }, TICKER_SCROLL_BOTTOM_PAUSE_MS);
    }
    return;
  }
  tickerState.scrollDelayTimer = setTimeout(() => {
    tickerState.scrollDelayTimer = null;
    tickerState.scrollTimer = setInterval(() => {
      wrapper.scrollTop += TICKER_SCROLL_STEP;
      if (wrapper.scrollTop + wrapper.clientHeight >= scrollHeight) {
        clearInterval(tickerState.scrollTimer);
        tickerState.scrollTimer = null;
        tickerState.scrollPauseTimer = setTimeout(() => {
          tickerState.scrollPauseTimer = null;
          advanceTickerSection();
        }, TICKER_SCROLL_BOTTOM_PAUSE_MS);
      }
    }, TICKER_SCROLL_INTERVAL_MS);
  }, TICKER_SCROLL_START_DELAY_MS);
}

function renderTickerSection(section) {
  const titleEl = document.getElementById('ticker-title');
  const listEl = document.getElementById('ticker-list');
  if (!titleEl || !listEl) return;
  stopTickerScroll();
  const key = toText(section.key).toLowerCase();
  if (key === 'recent_trades') {
    renderTickerTradesContent(section, titleEl, listEl);
  } else {
    renderTickerNetworthContent(section, titleEl, listEl);
  }
  prepareTickerLoop(listEl);
  startTickerScroll(listEl);
}

function renderTickerNetworthContent(section, titleEl, listEl) {
  const sectionTitle = toText(section.title);
  titleEl.innerHTML = '';
  const labelSpan = document.createElement('span');
  labelSpan.className = 'ticker-title-label';
  labelSpan.textContent = sectionTitle || '';
  const metaSpan = document.createElement('span');
  metaSpan.className = 'ticker-title-meta';
  metaSpan.textContent = toText(section.meta) || 'Est. Net Worth';
  titleEl.appendChild(labelSpan);
  titleEl.appendChild(metaSpan);
  listEl.innerHTML = '';
  const items = Array.isArray(section.items) ? section.items : [];
  items.forEach(it => {
    const li = document.createElement('li');
    li.className = 'ticker-list-item';
    const nm = toText(it.name);
    const nwRaw = toText(it.networth);
    const nw = normalizeMoney(nwRaw) || 'N/A';
    const partyRaw = toText(it.party);
    const party = partyRaw.toLowerCase();
    const partyDisplayRaw = toText(it.party_display || partyRaw);
    const partyDisplay = partyDisplayRaw ? partyDisplayRaw.trim().charAt(0).toUpperCase() : "";
    const stateDisplayRaw = toText(it.state_display || it.state);
    const stateDisplay = stateDisplayRaw ? stateDisplayRaw.trim().toUpperCase() : "";
    const rankVal = Number(it.rank) || "";

    const rankSpan = document.createElement('span');
    rankSpan.className = 'ticker-rank';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'ticker-name';
    const worthSpan = document.createElement('span');
    worthSpan.className = 'ticker-networth';
    if (party === 'democrat' || party === 'd') {
      nameSpan.classList.add('party-dem');
      worthSpan.classList.add('party-dem');
    } else if (party === 'republican' || party === 'r') {
      nameSpan.classList.add('party-rep');
      worthSpan.classList.add('party-rep');
    }
    rankSpan.textContent = rankVal ? `${rankVal}.` : '';

    const nameTextSpan = document.createElement('span');
    nameTextSpan.className = 'ticker-name-text';
    nameTextSpan.textContent = nm || PLACEHOLDER;
    nameSpan.appendChild(nameTextSpan);

    const suffixParts = [];
    if (partyDisplay) suffixParts.push(partyDisplay);
    if (stateDisplay) suffixParts.push(stateDisplay);
    const suffix = suffixParts.length ? ` (${suffixParts.join('-')})` : '';

    nameTextSpan.textContent = (nm || PLACEHOLDER) + suffix;

    worthSpan.textContent = nw;

    li.appendChild(rankSpan);
    li.appendChild(nameSpan);
    li.appendChild(worthSpan);
    listEl.appendChild(li);
  });
}

function partyClassFromValue(value) {
  const lower = toText(value).toLowerCase();
  if (lower === 'democrat' || lower === 'd') return 'party-dem';
  if (lower === 'republican' || lower === 'r') return 'party-rep';
  return '';
}

function splitNameLinesForTicker(fullName) {
  const text = toText(fullName);
  if (!text) return { first: PLACEHOLDER, last: '' };
  const parts = text.split(/\s+/).filter(Boolean);
  if (!parts.length) return { first: PLACEHOLDER, last: '' };
  if (parts.length === 1) {
    return { first: parts[0], last: '' };
  }
  const first = parts.shift();
  const last = parts.join(' ');
  return { first, last };
}

function buildTickerTradeRow(trade) {
  if (!trade || typeof trade !== 'object') return null;

  const li = document.createElement('li');
  li.className = 'ticker-list-item ticker-list-item--trade';

  const card = document.createElement('div');
  card.className = 'ticker-trade-card ticker-trade-card--split';
  li.appendChild(card);

  const nameBlock = document.createElement('div');
  nameBlock.className = 'ticker-trade-name-block ticker-name';
  const { first, last } = splitNameLinesForTicker(trade?.name);
  const partyClass = partyClassFromValue(trade?.party_display || trade?.party);
  if (partyClass) {
    nameBlock.classList.add(partyClass);
  }

  const firstLineEl = document.createElement('span');
  firstLineEl.className = 'ticker-trade-name-line ticker-trade-name-line--first ticker-name-text';
  firstLineEl.textContent = first || PLACEHOLDER;
  nameBlock.appendChild(firstLineEl);

  const lastLineEl = document.createElement('span');
  lastLineEl.className = 'ticker-trade-name-line ticker-trade-name-line--last ticker-name-text';
  lastLineEl.textContent = last || '\u00a0';
  nameBlock.appendChild(lastLineEl);

  card.appendChild(nameBlock);

  const detailCard = document.createElement('div');
  detailCard.className = 'ticker-trade-detail-card';
  card.appendChild(detailCard);

  const dateValue = fmtDate(
    trade?.traded ||
    trade?.transaction_date ||
    trade?.date ||
    trade?.trade_date
  );

  const fullDate = formatTradeDateFull(dateValue);

  const rangeText = formatRangeOrAmount(
    trade?.amount_range ||
    trade?.range ||
    trade?.Range ||
    trade?.range_display ||
    trade?.rangeText ||
    trade?.RangeDisplay,
    trade?.amount ||
    trade?.Amount ||
    trade?.trade_size_usd ||
    trade?.Trade_Size_USD ||
    trade?.Trade_Size_USD_num
  );
  const { low: rangeLow, high: rangeHigh } = splitRangeParts(rangeText);
  const lowerAmount = rangeLow || rangeText || PLACEHOLDER;
  const upperAmount = rangeHigh || rangeLow || rangeText || PLACEHOLDER;

  const tickerText = resolveTradeTicker(trade);
  const companyName = truncateCompanyName(resolveTradeAssetName(trade, tickerText));

  const { action, modifier, tone } = resolveTradeAction(trade);
  const actionLabel = action || PLACEHOLDER;

  const grid = document.createElement('div');
  grid.className = 'ticker-trade-grid';
  detailCard.appendChild(grid);

  const actionCell = createTickerGridCell(actionLabel, ['trade-col', 'trade-col--txn']);
  if (/buy/i.test(tone)) {
    actionCell.classList.add('trade-col--buy');
  }
  if (/(sell|sale)/i.test(tone)) {
    actionCell.classList.add('trade-col--sell');
  }
  grid.appendChild(actionCell);

  grid.appendChild(createTickerGridCell(lowerAmount, ['trade-col', 'trade-col--amount', 'ticker-trade-grid-cell--center']));

  grid.appendChild(createTickerGridCell(tickerText || PLACEHOLDER, ['trade-col', 'trade-col--asset', 'ticker-trade-ticker', 'ticker-trade-grid-cell--right']));

  grid.appendChild(createTickerGridCell(fullDate || PLACEHOLDER, ['ticker-trade-grid-cell--date']));

  grid.appendChild(createTickerGridCell(upperAmount, ['trade-col', 'trade-col--amount', 'ticker-trade-grid-cell--center']));

  grid.appendChild(createTickerGridCell(companyName || PLACEHOLDER, ['trade-asset-name', 'ticker-trade-company', 'ticker-trade-grid-cell--right']));

  return li;
}

function createTickerGridCell(text, extraClasses = []) {
  const cell = document.createElement('div');
  cell.className = 'ticker-trade-grid-cell';
  extraClasses.forEach(cls => cell.classList.add(cls));
  cell.textContent = text || PLACEHOLDER;
  return cell;
}

function formatTradeDateFull(dateValue) {
  if (dateValue && typeof dateValue === 'object' && dateValue.month) {
    // Convert month name to number
    const monthMap = {
      'jan': '01', 'january': '01',
      'feb': '02', 'february': '02',
      'mar': '03', 'march': '03',
      'apr': '04', 'april': '04',
      'may': '05',
      'jun': '06', 'june': '06',
      'jul': '07', 'july': '07',
      'aug': '08', 'august': '08',
      'sep': '09', 'sept': '09', 'september': '09',
      'oct': '10', 'october': '10',
      'nov': '11', 'november': '11',
      'dec': '12', 'december': '12'
    };

    const monthStr = String(dateValue.month).toLowerCase();
    const monthNum = monthMap[monthStr] || dateValue.month;
    const dayNum = dateValue.day ? String(dateValue.day).padStart(2, '0') : '01';
    const yearNum = dateValue.year || '';

    if (yearNum) {
      return `${monthNum}/${dayNum}/${yearNum}`;
    }
    return `${monthNum}/${dayNum}`;
  }
  return toText(dateValue) || PLACEHOLDER;
}

function splitRangeParts(label) {
  const text = toText(label);
  if (!text) return { low: '', high: '' };
  const parts = text.split(/\s*[-–—]\s*/).map(part => part.trim()).filter(Boolean);
  if (parts.length >= 2) {
    return { low: parts[0], high: parts[1] };
  }
  return { low: text, high: '' };
}

function renderTickerTradesContent(section, titleEl, listEl) {
  const sectionTitle = toText(section.title) || 'Recent Trades';
  titleEl.innerHTML = '';
  const labelSpan = document.createElement('span');
  labelSpan.className = 'ticker-title-label';
  labelSpan.textContent = sectionTitle;
  const metaSpan = document.createElement('span');
  metaSpan.className = 'ticker-title-meta';
  metaSpan.textContent = toText(section.meta) || 'Latest filings';
  titleEl.appendChild(labelSpan);
  titleEl.appendChild(metaSpan);
  listEl.innerHTML = '';
  const trades = Array.isArray(section.items) ? section.items : [];
  trades.forEach(trade => {
    const li = buildTickerTradeRow(trade);
    if (li) {
      listEl.appendChild(li);
    }
  });
}

function advanceTickerSection() {
  if (!tickerState.active) return;
  const sections = tickerState.sections || [];
  if (!sections.length) {
    return;
  }
  tickerState.sectionIndex = (tickerState.sectionIndex + 1) % sections.length;
  renderTickerSection(sections[tickerState.sectionIndex]);
}

function enterTickerMode(tickerData, hashOverride = null) {
  const grid = document.getElementById('card-container');
  const ticker = document.getElementById('ticker-mode');
  const networthSection = document.getElementById('networth-section');
  const header = document.querySelector('.overlay-header');
  const overlayEl = document.getElementById('overlay');
  if (!ticker || !grid) return;

  stopRotation();
  stopTickerTimers();

  grid.style.display = 'none';
  if (networthSection) networthSection.style.display = 'none';
  if (header) header.style.display = 'none';
  ticker.style.display = 'flex';
  if (overlayEl) overlayEl.classList.add('overlay--ticker-mode');

  const sections = Array.isArray(tickerData.sections) ? tickerData.sections : [];
  if (!sections.length) {
    exitTickerMode();
    return;
  }

  tickerState.sections = sections.slice();

  const hash = hashOverride || JSON.stringify(tickerData);
  if (tickerState.dataHash !== hash) {
    tickerState.sectionIndex = 0;
    tickerState.dataHash = hash;
  }

  tickerState.active = true;
  const headline = document.getElementById('speaker-name');
  if (headline) headline.textContent = toText(tickerData.title) || '';
  const bioSection = document.getElementById('bio-section');
  if (bioSection) {
    bioSection.style.display = 'none';
    bioSection.dataset.hasContent = 'false';
    const bioList = document.getElementById('bio-list');
    if (bioList) bioList.innerHTML = '';
  }

  renderTickerSection(sections[tickerState.sectionIndex]);
}

function formatRangeOrAmount(rangeRaw, amountRaw) {
  const preferred = toText(rangeRaw);
  const amount = toText(amountRaw);

  if (preferred && /[–—-]/.test(preferred)) {
    const parts = preferred.split(/[–—-]/).map(p => normalizeMoney(p));
    if (parts.filter(Boolean).length === parts.length && parts.length >= 2) {
      return `${parts[0]} - ${parts[1]}`;
    }
    return preferred;
  }

  const candidate = preferred || amount;
  if (!candidate) return "";

  const numeric = Number(candidate.replace(/[^0-9.]/g, '')) || 0;
  const lower = Math.floor(numeric);
  if (TRADE_RANGE_BY_LOWER.has(lower)) {
    return TRADE_RANGE_BY_LOWER.get(lower);
  }
  if (lower > 0) {
    const thresholds = [
      { min: 0, max: 1000, label: "$0 - $1,000" },
      { min: 1001, max: 15000, label: "$1,001 - $15,000" },
      { min: 15001, max: 50000, label: "$15,001 - $50,000" },
      { min: 50001, max: 100000, label: "$50,001 - $100,000" },
      { min: 100001, max: 250000, label: "$100,001 - $250,000" },
      { min: 250001, max: 500000, label: "$250,001 - $500,000" },
      { min: 500001, max: 1000000, label: "$500,001 - $1,000,000" },
      { min: 1000001, max: 5000000, label: "$1,000,001 - $5,000,000" },
      { min: 5000001, max: 25000000, label: "$5,000,001 - $25,000,000" },
      { min: 25000001, max: 50000000, label: "$25,000,001 - $50,000,000" },
      { min: 50000001, max: Number.POSITIVE_INFINITY, label: "$50,000,001+" },
    ];
    for (const bucket of thresholds) {
      if (lower >= bucket.min && lower <= bucket.max) {
        return bucket.label;
      }
    }
  }
  return normalizeMoney(candidate) || candidate;
}

function formatTradeLabel(tickerRaw, assetRaw) {
  const ticker = toText(tickerRaw);
  const asset = toText(assetRaw);
  const assetIsMeaningful = asset && /[A-Za-z]{3}/.test(asset) && !asset.includes('@');

  if (assetIsMeaningful) {
    return ticker ? `${asset} (${ticker})` : asset;
  }
  if (ticker) return ticker;
  return asset || PLACEHOLDER;
}

function formatTradeCountValue(raw) {
  const text = toText(raw);
  if (!text) return '';
  if (/trade/i.test(text)) return text;
  const normalized = text.replace(/,/g, '');
  const numeric = Number(normalized);
  if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
    const absolute = Math.abs(numeric);
    const display = Number.isInteger(numeric)
      ? numeric.toString()
      : Number(numeric.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')).toString();
    const label = absolute === 1 ? 'trade' : 'trades';
    return `${display} ${label}`;
  }
  return text;
}

const COMMITTEE_PHRASE_REPLACEMENTS = [
  [/\bHomeland Security\b/gi, 'Homeland Sec.'],
  [/\bForeign Affairs\b/gi, 'Foreign Aff.'],
  [/\bArmed Services\b/gi, 'Armed Svcs.'],
  [/\bEnergy and Commerce\b/gi, 'Energy & Com.'],
  [/\bEducation and the Workforce\b/gi, 'Edu. & Workforce'],
  [/\bEducation and Labor\b/gi, 'Edu. & Labor'],
  [/\bWays and Means\b/gi, 'Ways & Means'],
  [/\bAppropriations\b/gi, 'Approps'],
  [/\bTransportation\b/gi, 'Transp.'],
  [/\bInfrastructure\b/gi, 'Infra'],
  [/\bFinancial Services\b/gi, 'Fin. Svcs.'],
  [/\bFinancial Institutions\b/gi, 'Fin. Inst.'],
  [/\bSmall Business\b/gi, 'Small Biz'],
  [/\bNatural Resources\b/gi, 'Nat. Res.'],
  [/\bVeterans'? Affairs\b/gi, 'Vet. Aff.'],
  [/\bDepartments\b/gi, 'Depts.'],
  [/\bDepartment\b/gi, 'Dept.'],
  [/\bRelated Agencies\b/gi, 'Rel. Agencies'],
  [/\bScience,? Space,? and Technology\b/gi, 'Sci. & Tech.'],
  [/\bPublic Works\b/gi, 'Pub. Works'],
  [/\bEconomic Development\b/gi, 'Econ. Dev.'],
  [/\bEconomic Opportunity\b/gi, 'Econ. Opp.'],
  [/\bOversight and Reform\b/gi, 'Ovrs. & Ref.'],
  [/\bOversight and Accountability\b/gi, 'Ovrs. & Acct.'],
  [/\bStrategic Forces\b/gi, 'Strat. Forces'],
  [/\bCivil Rights and Liberties\b/gi, 'Civil Rights & Lib.'],
  [/\bEnvironment\b/gi, 'Env.'],
  [/\bInvestigations\b/gi, 'Invest.'],
  [/\bHealthcare\b/gi, 'Health'],
  [/\bInnovation\b/gi, 'Innov.'],
  [/\bTechnology\b/gi, 'Tech.']
];

const COMMITTEE_WORD_MAP = {
  'services': 'Svcs.',
  'service': 'Svc.',
  'government': 'Govt.',
  'department': 'Dept.',
  'departments': 'Depts.',
  'technology': 'Tech.',
  'technologies': 'Tech.',
  'communications': 'Comms.',
  'communication': 'Comm.',
  'economic': 'Econ.',
  'international': 'Intl.',
  'security': 'Sec.',
  'intelligence': 'Intel.',
  'science': 'Sci.',
  'financial': 'Fin.',
  'resources': 'Res.',
  'resource': 'Res.',
  'management': 'Mgmt.',
  'administration': 'Admin.',
  'innovation': 'Innov.',
  'housing': 'Hsg.',
  'urban': 'Urb.',
  'community': 'Comm.',
  'communities': 'Comm.',
  'investment': 'Invest.',
  'investments': 'Invest.',
  'investigation': 'Invest.',
  'investigations': 'Invest.',
  'affairs': 'Aff.',
  'industries': 'Inds.',
  'industry': 'Ind.',
  'relation': 'Rel.',
  'relations': 'Rel.',
  'related': 'Rel.',
  'transportation': 'Transp.',
  'infrastructure': 'Infra',
  'environment': 'Env.',
  'environmental': 'Env.',
  'health': 'Hlth.',
  'human': 'Hum.',
  'veterans': 'Vets.',
  'business': 'Biz.',
  'judiciary': 'Judic.',
  'commerce': 'Com.',
  'banking': 'Bank.',
  'permanent': 'Perm.',
  'special': 'Spec.',
  'joint': 'Jt.',
  'subcommittee': 'Subcmte'
};

function applyWordMap(str, dictionary) {
  return str.replace(/\b([A-Za-z][A-Za-z']*)\b/g, (token) => {
    const key = token.toLowerCase();
    return dictionary[key] || token;
  });
}

function abbreviateCommittee(name) {
  if (!name) return "";
  let s = String(name).trim();
  // Light cleanup: drop redundant prefixes but keep full wording
  s = s.replace(/^\s*(Committee on|Committee)\s+/i, '');
  s = s.replace(/^\s*(House|Senate)\s+/i, '');
  s = s.replace(/\s+/g, ' ').trim();
  s = s.replace(/^the\b/i, 'The');
  return s;
}

function abbreviateSub(name) {
  if (!name) return "";
  let s = String(name).trim();
  s = s.replace(/^\s*(Subcommittee|Subcmte)\s*(on)?\s*/i, '');
  s = s.replace(/^[-\u2013\u2014:\s]+/, '');
  s = s.replace(/\s+/g, ' ').trim();

  const PHRASE_RELAX = [
    [/Investigations?\s+and\s+Oversight/gi, 'Oversight & Investigations'],
    [/Oversight\s+and\s+Investigations/gi, 'Oversight & Investigations'],
    [/Research\s+and\s+Technology/gi, 'Research & Technology'],
    [/Science,\s*Space,\s*and\s*Technology/gi, 'Science, Space & Technology'],
    [/Labor.*Health.*Human.*Services.*Education/gi, 'Labor, Health & Education'],
    [/Government\s+Operations\s+and\s+Innovation/gi, 'Government Operations & Innovation'],
    [/Economic\s+Opportunity/gi, 'Economic Opportunity']
  ];
  for (const [re, to] of PHRASE_RELAX) {
    s = s.replace(re, to);
  }

  const MAX_LEN = 64;
  const fallbackReplacements = [
    [/\bUnited States\b/gi, 'U.S.'],
    [/\bDepartment\b/gi, 'Dept.'],
    [/\bDepartments\b/gi, 'Depts.'],
    [/\bAdministration\b/gi, 'Admin.'],
    [/\bGovernment\b/gi, 'Govt.'],
    [/\bInternational\b/gi, 'Intl.'],
    [/\bTransportation\b/gi, 'Transport'],
    [/\bInfrastructure\b/gi, 'Infra.'],
    [/\bDevelopment\b/gi, 'Dev.'],
    [/\bCommunity\b/gi, 'Community'],
    [/\bNational\b/gi, 'National'],
    [/\bSecurity\b/gi, 'Security'],
    [/\bServices\b/gi, 'Services'],
  ];

  for (const [pattern, replacement] of fallbackReplacements) {
    if (s.length <= MAX_LEN) break;
    s = s.replace(pattern, replacement);
  }

  if (s.length > MAX_LEN) {
    s = s.replace(/\s+and\s+/gi, ' & ');
  }

  return s.replace(/\s+/g, ' ').trim();
}

const ROTATABLE_CARD_IDS = ['bio-section','committees-section','holdings-section','sectors-section','trades-section','donors-section','industries-section'];
const ROTATION_INTERVAL_MS = 8000; // 10 seconds per card (industry standard)
const ROTATION_DELAY_MS = 5000; // Initial delay before first scroll
const SCROLL_DURATION_MS = 2000; // 2 seconds for smooth scroll (industry standard)
const cardRotation = {
  container: null,
  cards: [],
  currentIndex: 0, // simplified: just track which card we're showing
  timer: null,
  delayTimer: null,
  animationFrame: null,
  isScrolling: false,
  lastDataHash: "", // Track data changes to prevent scroll interruption
  cardFingerprint: "" // Track which cards are active to avoid unnecessary clone rebuilds
};
let loopCloneIdCounter = 0;

function stopRotationTimer() {
  if (cardRotation.timer) {
    clearInterval(cardRotation.timer);
    cardRotation.timer = null;
  }
}

function stopDelayTimer() {
  if (cardRotation.delayTimer) {
    clearTimeout(cardRotation.delayTimer);
    cardRotation.delayTimer = null;
  }
}

function stopRotation() {
  stopRotationTimer();
  stopDelayTimer();
  if (cardRotation.animationFrame) {
    cancelAnimationFrame(cardRotation.animationFrame);
    cardRotation.animationFrame = null;
  }
  cardRotation.cards = [];
  cardRotation.currentIndex = 0;
  cardRotation.isScrolling = false;
}

function startRotationTimer() {
  if (!cardRotation.timer && cardRotation.cards.length > 1) {
    cardRotation.timer = setInterval(rotateCardSet, ROTATION_INTERVAL_MS);
  }
}

function scheduleRotationTimer() {
  stopRotationTimer();
  stopDelayTimer();
  if (cardRotation.cards.length > 1) {
    cardRotation.delayTimer = setTimeout(() => {
      cardRotation.delayTimer = null;
      rotateCardSet();
      startRotationTimer();
    }, ROTATION_DELAY_MS);
  }
}

function smoothScrollTo(targetScrollTop, duration = SCROLL_DURATION_MS, onComplete = null) {
  const container = cardRotation.container;
  if (!container) return;
  
  const start = container.scrollTop;
  const distance = targetScrollTop - start;
  
  if (cardRotation.animationFrame) {
    cancelAnimationFrame(cardRotation.animationFrame);
    cardRotation.animationFrame = null;
  }
  
  if (duration === 0 || Math.abs(distance) < 1) {
    container.scrollTop = targetScrollTop;
    cardRotation.isScrolling = false;
    if (onComplete) onComplete();
    return;
  }
  
  cardRotation.isScrolling = true;
  let startTime = null;
  // Smoother easing function - ease-in-out cubic for more natural motion
  const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  
  function step(timestamp) {
    if (startTime === null) startTime = timestamp;
    const elapsed = timestamp - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = easeInOutCubic(progress);
    container.scrollTop = start + distance * eased;
    
    if (progress < 1) {
      cardRotation.animationFrame = requestAnimationFrame(step);
    } else {
      cardRotation.animationFrame = null;
      cardRotation.isScrolling = false;
      if (onComplete) onComplete();
    }
  }
  cardRotation.animationFrame = requestAnimationFrame(step);
}

function getCardScrollPosition(cardOrIndex) {
  const container = cardRotation.container;
  if (!container) return 0;

  const card = typeof cardOrIndex === 'number'
    ? cardRotation.cards[cardOrIndex]
    : cardOrIndex;
  if (!card) return 0;

  const styles = window.getComputedStyle(container);
  const paddingTop = parseFloat(styles.getPropertyValue('padding-top')) || 0;

  return card.offsetTop - paddingTop;
}

function reconcileCardRotation() {
  const container = document.getElementById('card-container');
  if (!container) return;
  cardRotation.container = container;

  // Collect cards in the order they appear in ROTATABLE_CARD_IDS (display order)
  const activeCards = [];
  for (const id of ROTATABLE_CARD_IDS) {
    const el = document.getElementById(id);
    if (el && el.dataset.hasContent === 'true') {
      activeCards.push(el);
    }
  }

  // Generate fingerprint of active cards to detect real changes
  const cardFingerprint = activeCards.map(c => c.id).join(',');
  const previousFingerprint = cardRotation.cardFingerprint || '';
  const cardsChanged = cardFingerprint !== previousFingerprint;

  const wasAnimating = cardRotation.isScrolling;
  const previousCount = cardRotation.cards.length;

  // If we're in the middle of a scroll animation, DON'T touch clones
  // This prevents the glitch when scrolling to a clone that gets deleted
  if (wasAnimating && !cardsChanged) {
    // Cards haven't changed, just let the animation complete
    return;
  }

  // Now safe to remove and rebuild clones
  const existingCloneWrappers = container.querySelectorAll('[data-loop-clone-wrapper="true"]');
  existingCloneWrappers.forEach(wrapper => wrapper.remove());

  // Also remove any orphaned clones (without wrappers) from older code versions
  const existingClones = container.querySelectorAll('[data-loop-clone="true"]');
  existingClones.forEach(clone => clone.remove());

  if (!activeCards.length) {
    stopRotation();
    cardRotation.cardFingerprint = '';
    container.scrollTo({ top: 0, behavior: 'auto' });
    return;
  }

  cardRotation.cards = activeCards;
  cardRotation.cardFingerprint = cardFingerprint;

  // Reset index if out of bounds
  if (cardRotation.currentIndex >= cardRotation.cards.length) {
    cardRotation.currentIndex = 0;
  }

  if (cardRotation.cards.length <= 1) {
    stopRotation();
    // Only scroll if not currently animating
    if (!wasAnimating) {
      container.scrollTo({ top: getCardScrollPosition(0), behavior: 'auto' });
    }
    return;
  }

  // Create clones for infinite loop effect (only if we have multiple cards)
  if (cardRotation.cards.length > 1) {
    const fragment = document.createDocumentFragment();

    activeCards.forEach((card, index) => {
      const clone = card.cloneNode(true);
      loopCloneIdCounter++;

      // Mark as clone and preserve reference to original ID
      clone.dataset.loopClone = 'true';
      clone.dataset.originalId = card.id;
      clone.dataset.cloneIndex = String(index);

      // Reset problematic inline sizing/visibility styles without losing other formatting
      if (clone.style) {
        ['height', 'minHeight', 'maxHeight', 'transform', 'opacity', 'display'].forEach(prop => {
          clone.style[prop] = '';
        });
      }

      // Ensure data-has-content is preserved (CRITICAL for rotation logic)
      clone.dataset.hasContent = 'true';

      // Generate unique ID for the cloned section
      const sectionCloneId = `loop-clone-${loopCloneIdCounter}-${index}`;
      clone.id = sectionCloneId;

      // Recursively update all child element IDs and set data-original-id
      const elementsWithIds = clone.querySelectorAll('[id]');
      elementsWithIds.forEach((el) => {
        const originalId = el.id;
        el.dataset.originalId = originalId;
        // Create new unique ID
        el.id = `${sectionCloneId}-${originalId}`;

        // Only remove display:none inline styles, preserve other layout styles
        if (el.style.display === 'none') {
          el.style.display = '';
        }
      });

      // CRITICAL: Wrap clone in a .column div to match the original structure
      // This preserves the grid layout that sections need
      const columnWrapper = document.createElement('div');
      columnWrapper.className = 'column';
      columnWrapper.dataset.loopCloneWrapper = 'true';
      columnWrapper.appendChild(clone);

      fragment.appendChild(columnWrapper);
    });

    // Append all clones at once to minimize reflows
    container.appendChild(fragment);
  }

  // Synchronize clone content with originals to ensure exact match
  syncCloneContent();

  const finalizeRotationSetup = () => {
    // DON'T interrupt ongoing scroll animations - let them complete
    if (wasAnimating) {
      // Just restart the timer if needed, don't touch scroll position
      const hadTimer = Boolean(cardRotation.delayTimer || cardRotation.timer);
      if (!hadTimer || previousCount !== cardRotation.cards.length) {
        scheduleRotationTimer();
      }
      return;
    }

    // Only adjust scroll position if we're NOT animating
    container.scrollTo({ top: getCardScrollPosition(cardRotation.currentIndex), behavior: 'auto' });

    // Only restart timer if cards changed significantly or we weren't already running
    const hadTimer = Boolean(cardRotation.delayTimer || cardRotation.timer);
    if (!hadTimer || previousCount !== cardRotation.cards.length) {
      scheduleRotationTimer();
    }
  };

  if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
    window.requestAnimationFrame(finalizeRotationSetup);
  } else {
    finalizeRotationSetup();
  }
}

// Ensure clone innerHTML exactly matches original to prevent visual differences
function syncCloneContent() {
  const container = cardRotation.container;
  if (!container) return;

  const clones = container.querySelectorAll('[data-loop-clone="true"]');
  clones.forEach(clone => {
    const originalId = clone.dataset.originalId;
    if (!originalId) return;

    const original = document.getElementById(originalId);
    if (!original) return;

    // Compare and sync the inner content of key child elements
    // This ensures lists, text content, etc. are identical
    const originalLists = original.querySelectorAll('ul, .trade-list');
    const cloneLists = clone.querySelectorAll('ul, .trade-list');

    originalLists.forEach((origList, i) => {
      const cloneList = cloneLists[i];
      if (cloneList && origList.innerHTML !== cloneList.innerHTML) {
        // Rebuild the clone's list content from the original
        cloneList.innerHTML = origList.innerHTML;
        // Re-apply data-original-id to any new elements with IDs
        const newElements = cloneList.querySelectorAll('[id]');
        newElements.forEach(el => {
          const origElId = el.id;
          el.dataset.originalId = origElId;
          el.id = `${clone.id}-${origElId}`;
        });
      }
    });
  });
}

function rotateCardSet() {
  const container = cardRotation.container;
  if (!container || !cardRotation.cards.length || cardRotation.isScrolling) {
    return;
  }

  if (cardRotation.cards.length === 1) {
    stopRotation();
    container.scrollTo({ top: getCardScrollPosition(0), behavior: 'auto' });
    return;
  }

  const nextIndex = cardRotation.currentIndex + 1;
  const isWrapping = nextIndex >= cardRotation.cards.length;

  if (isWrapping) {
    // Infinite loop: scroll to the first CLONED card
    const firstClone = container.querySelector('[data-loop-clone="true"][data-clone-index="0"]');
    if (!firstClone) {
      // Fallback if no clones exist - shouldn't happen but safe
      const firstCard = cardRotation.cards[0];
      const targetPos = getCardScrollPosition(firstCard);
      smoothScrollTo(targetPos, SCROLL_DURATION_MS, () => {
        cardRotation.currentIndex = 0;
        startRotationTimer();
      });
      return;
    }

    // Scroll to the cloned first card
    // Since clones are wrapped in .column divs, we need to account for the wrapper's offset
    const styles = window.getComputedStyle(container);
    const paddingTop = parseFloat(styles.getPropertyValue('padding-top')) || 0;

    // Get the column wrapper parent
    const columnWrapper = firstClone.parentElement;
    let cloneTargetPos = 0;

    if (columnWrapper && columnWrapper.dataset.loopCloneWrapper === 'true') {
      // Clone is wrapped - use wrapper's offsetTop
      cloneTargetPos = columnWrapper.offsetTop - paddingTop;
    } else {
      // Fallback for unwrapped clones (shouldn't happen with current code)
      cloneTargetPos = firstClone.offsetTop - paddingTop;
    }

    smoothScrollTo(cloneTargetPos, SCROLL_DURATION_MS, () => {
      // After reaching the clone, instantly jump back to the original first card
      const firstCard = cardRotation.cards[0];
      const originalPos = getCardScrollPosition(firstCard);
      container.scrollTop = originalPos;
      cardRotation.currentIndex = 0;
      startRotationTimer();
    });
  } else {
    // Normal forward scroll to next card
    const nextCard = cardRotation.cards[nextIndex];
    if (!nextCard) {
      startRotationTimer();
      return;
    }

    const targetPos = getCardScrollPosition(nextCard);
    smoothScrollTo(targetPos, SCROLL_DURATION_MS, () => {
      cardRotation.currentIndex = nextIndex;
      startRotationTimer();
    });
  }
}

// avoid overlapping fetches
let inflight = false;
let lastSpeakerName = null;
let consecutiveErrors = 0;
const MAX_ERRORS = 3;
const DEBUG = false; // Set to true only when debugging
let lastDataFingerprint = ""; // Track significant data changes

function renderSpeakerName(data) {
  const speakerNameEl = document.getElementById('speaker-name');
  if (!speakerNameEl) {
    return;
  }
  const fallbackName = 'No speaker detected';
  const providedName = toText(data?.name);
  const rawName = providedName || fallbackName;
  const segments = rawName.split(/\n+/).map(part => part.trim()).filter(Boolean);

  speakerNameEl.textContent = '';
  speakerNameEl.classList.remove('speaker-name--with-meta');
  if (!segments.length) {
    speakerNameEl.textContent = fallbackName;
    return;
  }

  const [primary, ...rest] = segments;
  const primaryEl = document.createElement(providedName ? 'strong' : 'span');
  primaryEl.textContent = primary;
  if (!providedName) {
    primaryEl.className = 'speaker-name__primary';
  }
  speakerNameEl.appendChild(primaryEl);

  rest.forEach(segment => {
    if (!segment) return;
    const metaSpan = document.createElement('span');
    metaSpan.className = 'speaker-name__meta';
    metaSpan.textContent = segment;
    speakerNameEl.appendChild(metaSpan);
  });
  if (rest.length) {
    speakerNameEl.classList.add('speaker-name--with-meta');
  }
}

function renderNetworth(data) {
  const networthEl = document.getElementById('networth');
  if (!networthEl) {
    return;
  }
  networthEl.textContent = preferNetWorth(data);
}

function resolveBioItems(data) {
  const rawItems = Array.isArray(data?.bio_card?.items) ? data.bio_card.items : [];
  const resolved = [];

  rawItems.forEach(item => {
    if (item && typeof item === 'object' && !Array.isArray(item)) {
      const label = toText(item.label);
      const value = toText(item.value);
      const text = toText(item.text);
      if (label && value) {
        resolved.push({ type: 'pair', label, value });
      } else if (label && !value && label.trim().toLowerCase() === 'education') {
        resolved.push({ type: 'pair', label, value: '' });
      } else if (text) {
        resolved.push({ type: 'text', text });
      } else if (value) {
        resolved.push({ type: 'text', text: value });
      }
    } else {
      const fallback = toText(item);
      if (fallback) {
        resolved.push({ type: 'text', text: fallback });
      }
    }
  });

  if (!resolved.length) {
    const dob = toText(data?.dob);
    const age = toText(data?.age);
    const tenure = toText(data?.tenure_pretty);
    if (dob) {
      const prettyDob = fmtDateLongMDY(dob) || dob;
      resolved.push({ type: 'pair', label: 'Date of Birth', value: prettyDob });
    }
    if (age) {
      resolved.push({ type: 'pair', label: 'Age', value: age });
    }
    if (tenure) {
      const normalizedTenure = formatTenurePretty(tenure) || tenure;
      resolved.push({ type: 'pair', label: 'Time in Office', value: normalizedTenure });
    }
  }

  return resolved;
}

function renderBioSection(data) {
  const bioSection = document.getElementById('bio-section');
  const bioList = document.getElementById('bio-list');
  if (!bioSection || !bioList) {
    return;
  }

  bioList.innerHTML = '';
  const resolvedItems = resolveBioItems(data);

  let i = 0;
  while (i < resolvedItems.length) {
    const entry = resolvedItems[i];
    if (entry.type === 'pair' && entry.label) {
      const label = entry.label.trim();
      if (!label) {
        i += 1;
        continue;
      }
      if (label.toLowerCase() === 'education') {
        const { degrees, nextIndex } = collectEducationEntries(resolvedItems, i);
        if (degrees.length) {
          const educationLi = document.createElement('li');
          educationLi.className = 'education-label bio-pair';
          const strongEl = document.createElement('strong');
          strongEl.textContent = 'Education:';
          strongEl.classList.add('bio-label');
          educationLi.appendChild(strongEl);

          const firstDegree = degrees[0];
          if (firstDegree) {
            const inlineSpan = document.createElement('span');
            inlineSpan.className = 'education-degree-inline accent-text bio-value';
            inlineSpan.textContent = firstDegree;
            educationLi.appendChild(inlineSpan);
          }

          bioList.appendChild(educationLi);

          degrees.slice(1).forEach(degree => {
            const li = document.createElement('li');
            li.className = 'bio-pair education-degree-row';

            const spacer = document.createElement('strong');
            spacer.classList.add('bio-label', 'education-label-spacer');
            spacer.textContent = '';
            li.appendChild(spacer);

            const span = document.createElement('span');
            span.classList.add('accent-text', 'bio-value', 'education-degree-value');
            span.textContent = degree;
            li.appendChild(span);

            bioList.appendChild(li);
          });
        }
        i = nextIndex;
        continue;
      }

      const value = toText(entry.value);
      if (value) {
        const li = document.createElement('li');
        li.classList.add('bio-pair');
        const strongEl = document.createElement('strong');
        strongEl.classList.add('bio-label');
        strongEl.textContent = `${label}: `;
        li.appendChild(strongEl);

        const span = document.createElement('span');
        span.classList.add('accent-text', 'bio-value');
        span.textContent = value;
        li.appendChild(span);

        bioList.appendChild(li);
      }
    } else if (entry.type === 'text' && entry.text) {
      const li = document.createElement('li');
      li.textContent = entry.text;
      bioList.appendChild(li);
    }
    i += 1;
  }

  setSectionState(bioSection, bioList.children.length > 0);
}

function renderDonorSection(data) {
  const section = document.getElementById('donors-section');
  const list = document.getElementById('donor-list');
  const eyebrow = document.getElementById('donors-eyebrow');
  const period = document.getElementById('donors-period');
  if (!section || !list || !eyebrow || !period) {
    return;
  }

  ensureKvList(list);
  list.innerHTML = '';
  const donors = Array.isArray(data?.top_donors) ? data.top_donors.slice(0, 5) : [];
  donors.forEach(donor => {
    const li = document.createElement('li');
    const nameSpan = document.createElement('span');
    nameSpan.className = 'kv-label';
    nameSpan.textContent = textOrPlaceholder(donor?.name);
    li.appendChild(nameSpan);

    const amountSpan = document.createElement('span');
    amountSpan.className = 'kv-amount accent-text';
    amountSpan.textContent = moneyOrPlaceholder(donor?.amount);
    li.appendChild(amountSpan);

    list.appendChild(li);
  });

  eyebrow.textContent = 'Top Campaign Contributors';
  const periodText = toText(data?.periods?.contributors);
  period.textContent = periodText;
  period.style.display = periodText ? 'block' : 'none';

  setSectionState(section, list.children.length > 0);
}

function renderIndustrySection(data) {
  const section = document.getElementById('industries-section');
  const list = document.getElementById('industry-list');
  const eyebrow = document.getElementById('industries-eyebrow');
  const period = document.getElementById('industries-period');
  if (!section || !list || !eyebrow || !period) {
    return;
  }

  ensureKvList(list);
  list.innerHTML = '';
  const industries = Array.isArray(data?.top_industries) ? data.top_industries.slice(0, 5) : [];
  industries.forEach(item => {
    const li = document.createElement('li');
    const nameSpan = document.createElement('span');
    nameSpan.className = 'kv-label';
    const name = toText(item?.name);
    nameSpan.textContent = name && name.toLowerCase() === 'retired' ? 'Retired Donors' : (name || PLACEHOLDER);
    li.appendChild(nameSpan);

    const amountSpan = document.createElement('span');
    amountSpan.className = 'kv-amount accent-text';
    amountSpan.textContent = moneyOrPlaceholder(item?.amount);
    li.appendChild(amountSpan);

    list.appendChild(li);
  });

  eyebrow.textContent = 'Top Donor Industries';
  const periodText = toText(data?.periods?.industries);
  period.textContent = periodText;
  period.style.display = periodText ? 'block' : 'none';

  setSectionState(section, list.children.length > 0);
}

function renderHoldingsSection(data) {
  const section = document.getElementById('holdings-section');
  const list = document.getElementById('holdings-list');
  if (!section || !list) {
    return;
  }

  ensureKvList(list);
  list.innerHTML = '';
  const holdings = Array.isArray(data?.top_holdings) ? data.top_holdings.slice(0, 5) : [];
  holdings.forEach(item => {
    const li = document.createElement('li');
    const nameSpan = document.createElement('span');
    nameSpan.className = 'kv-label';
    nameSpan.textContent = textOrPlaceholder(item?.name);
    li.appendChild(nameSpan);

    const amountSpan = document.createElement('span');
    amountSpan.className = 'kv-amount accent-text';
    amountSpan.textContent = moneyOrPlaceholder(item?.amount);
    li.appendChild(amountSpan);

    list.appendChild(li);
  });

  setSectionState(section, list.children.length > 0);
}

function renderSectorsSection(data) {
  const section = document.getElementById('sectors-section');
  const list = document.getElementById('sectors-list');
  if (!section || !list) {
    return;
  }

  list.innerHTML = '';
  const sectorsRaw = Array.isArray(data?.top_traded_sectors) ? data.top_traded_sectors : [];
  const sectors = sectorsRaw
    .filter(entry => {
      const sectorName = entry ? (entry.sector || entry.name) : '';
      return !!toText(sectorName);
    })
    .slice(0, 5);

  if (sectors.length < 1) {
    setSectionState(section, false);
    return;
  }

  sectors.forEach(entry => {
    const li = document.createElement('li');
    const sectorName = entry ? (entry.sector || entry.name) : '';
    li.textContent = textOrPlaceholder(sectorName);
    list.appendChild(li);
  });

  setSectionState(section, list.children.length > 0);
}

function renderCommitteesSection(data) {
  const section = document.getElementById('committees-section');
  const list = document.getElementById('committee-list');
  if (!section || !list) {
    return;
  }

  list.innerHTML = '';
  const committees = Array.isArray(data?.committees) ? data.committees : [];
  const grouped = new Map();

  committees.forEach(entry => {
    const committeeName = abbreviateCommittee(toText(entry?.committee));
    if (!committeeName) {
      return;
    }
    let subcommittee = abbreviateSub(toText(entry?.subcommittee));
    if (/^(?:Subcmte)?\s*$/.test(subcommittee)) {
      subcommittee = '';
    }
    const bucket = grouped.get(committeeName) || [];
    if (subcommittee) {
      bucket.push(subcommittee);
    }
    grouped.set(committeeName, bucket);
  });

  grouped.forEach((subsRaw, committeeName) => {
    const subs = [...new Set(subsRaw.filter(Boolean))];
    const hasSubs = subs.length > 0;
    const li = document.createElement('li');
    if (hasSubs) {
      li.classList.add('committee-has-sublist');
    }
    const line = document.createElement('span');
    line.classList.add('list-heading', 'committee-line');
    line.textContent = committeeName;
    li.appendChild(line);
    if (subs.length) {
      const subList = document.createElement('ul');
      subList.classList.add('committee-sublist');
      subs.forEach(sub => {
        const subLi = document.createElement('li');
        subLi.classList.add('committee-subitem');
        const textSpan = document.createElement('span');
        textSpan.textContent = sub;

        subLi.appendChild(textSpan);
        subList.appendChild(subLi);
      });
      li.appendChild(subList);
    }
    list.appendChild(li);
  });

  setSectionState(section, grouped.size > 0);
}

function resolveTradeTicker(trade) {
  const primary = toText(trade?.ticker || trade?.Ticker || trade?.symbol);
  return primary ? primary.toUpperCase() : PLACEHOLDER;
}

function resolveTradeAssetName(trade, tickerText) {
  const candidates = [
    trade?.company_clean,
    trade?.description,
    trade?.Description,
    trade?.company,
    trade?.Company,
    trade?.asset_name,
    trade?.AssetName,
    trade?.asset,
    trade?.security_name,
    trade?.SecurityName,
    trade?.security,
    trade?.Security,
  ];
  for (const candidate of candidates) {
    const value = toText(candidate);
    if (value && value.toLowerCase() !== tickerText.toLowerCase()) {
      return value;
    }
  }

  // Fallback: lookup ticker in company ticker map
  if (tickerText && companyTickerMap[tickerText.toUpperCase()]) {
    return companyTickerMap[tickerText.toUpperCase()];
  }

  return '';
}

/**
 * Truncate long company names to avoid overflow.
 * Strategy: Keep first 1-2 words if name is too long.
 * @param {string} companyName - Full company name
 * @param {number} maxChars - Maximum character length (default 20)
 * @returns {string} - Truncated company name
 */
function truncateCompanyName(companyName, maxChars = 20) {
  if (!companyName || companyName.length <= maxChars) {
    return companyName;
  }

  // Common suffixes to remove if name is too long
  const suffixes = [
    ' Inc', ' Inc.', ' Corp', ' Corp.', ' Corporation',
    ' LLC', ' L.L.C.', ' Ltd', ' Ltd.', ' Limited',
    ' Co', ' Co.', ' Company', ' Group', ' Holdings',
    ' Plc', ' PLC', ' AG', ' SA', ' SE', ' NV'
  ];

  let truncated = companyName;

  // First try: remove common suffixes
  for (const suffix of suffixes) {
    if (truncated.endsWith(suffix)) {
      truncated = truncated.slice(0, -suffix.length).trim();
      if (truncated.length <= maxChars) {
        return truncated;
      }
    }
  }

  // Second try: keep only first 1-2 words
  const words = truncated.split(' ').filter(w => w.length > 0);

  if (words.length >= 1 && words[0].length <= maxChars) {
    // Try first word only
    if (words[0].length >= maxChars * 0.6) {
      return words[0];
    }

    // Try first two words
    if (words.length >= 2) {
      const twoWords = words.slice(0, 2).join(' ');
      if (twoWords.length <= maxChars) {
        return twoWords;
      }
    }

    // Fall back to first word
    return words[0];
  }

  // Last resort: hard truncate at maxChars
  return truncated.substring(0, maxChars).trim();
}

function resolveTradeAction(trade) {
  const raw = toText(trade?.transaction || trade?.Transaction);
  if (!raw) {
    return { action: '', modifier: '', tone: '' };
  }
  const tone = normalizeTxn(raw) || raw;
  const lower = raw.toLowerCase();
  let action = '';
  if (lower.includes('buy') || lower.includes('purchase')) {
    action = 'Buy';
  } else if (lower.includes('sell') || lower.includes('sale')) {
    action = 'Sell';
  } else {
    action = raw;
  }
  let modifier = '';
  if (lower.includes('full')) {
    modifier = '(full)';
  } else if (lower.includes('partial')) {
    modifier = '(partial)';
  }
  return { action, modifier, tone };
}

function createTradeRowElement(trade) {
  if (!trade || typeof trade !== 'object') {
    return null;
  }
  const row = document.createElement('div');
  row.classList.add('trade-item');

  const assetBlock = document.createElement('span');
  assetBlock.className = 'trade-col trade-col--asset-block';
  const tickerSpan = document.createElement('span');
  tickerSpan.className = 'trade-col--asset';
  const tickerText = resolveTradeTicker(trade);
  tickerSpan.textContent = tickerText;
  assetBlock.appendChild(tickerSpan);

  const assetName = resolveTradeAssetName(trade, tickerText);
  if (assetName) {
    const assetNameSpan = document.createElement('span');
    assetNameSpan.className = 'trade-asset-name';
    assetNameSpan.textContent = truncateCompanyName(assetName);
    assetBlock.appendChild(assetNameSpan);
  }

  const txnBlock = document.createElement('span');
  txnBlock.className = 'trade-col trade-col--txn-block';

  const txnSpan = document.createElement('span');
  txnSpan.className = 'trade-col trade-col--txn';
  const { action, modifier, tone } = resolveTradeAction(trade);
  txnSpan.textContent = action || PLACEHOLDER;
  if (/buy/i.test(tone)) {
    txnSpan.classList.add('trade-col--buy');
  }
  if (/(sell|sale)/i.test(tone)) {
    txnSpan.classList.add('trade-col--sell');
  }
  if (modifier) {
    const modifierSpan = document.createElement('span');
    modifierSpan.className = 'trade-txn-modifier';
    modifierSpan.textContent = modifier;
    txnSpan.appendChild(document.createTextNode(' '));
    txnSpan.appendChild(modifierSpan);
  }

  const rangeText = toText(
    trade?.amount_range ||
    trade?.range ||
    trade?.Range ||
    trade?.range_display ||
    trade?.rangeText ||
    trade?.RangeDisplay
  );
  const amountText = toText(
    trade?.amount ||
    trade?.Amount ||
    trade?.trade_size_usd ||
    trade?.Trade_Size_USD ||
    trade?.Trade_Size_USD_num
  );
  const amountSpan = document.createElement('span');
  amountSpan.className = 'trade-col trade-col--amount';
  const formattedAmount = formatRangeOrAmount(rangeText, amountText);
  amountSpan.textContent = formattedAmount || PLACEHOLDER;

  txnBlock.appendChild(txnSpan);
  txnBlock.appendChild(amountSpan);

  const dateSpan = document.createElement('span');
  dateSpan.className = 'trade-col trade-col--date';
  const dateValue = fmtDate(
    trade?.traded ||
    trade?.transaction_date ||
    trade?.date ||
    trade?.trade_date
  );
  if (dateValue && typeof dateValue === 'object' && dateValue.month && dateValue.year) {
    const monthSpan = document.createElement('span');
    monthSpan.className = 'trade-date-month';
    monthSpan.textContent = dateValue.month;
    const yearSpan = document.createElement('span');
    yearSpan.className = 'trade-date-year';
    yearSpan.textContent = dateValue.year;
    dateSpan.appendChild(monthSpan);
    dateSpan.appendChild(yearSpan);
  } else {
    dateSpan.textContent = dateValue || PLACEHOLDER;
  }

  row.appendChild(assetBlock);
  row.appendChild(txnBlock);
  row.appendChild(dateSpan);
  return row;
}

function renderTradesSection(data) {
  const section = document.getElementById('trades-section');
  const list = document.getElementById('trades-list');
  if (!section || !list) {
    return;
  }

  list.innerHTML = '';
  const tradesRaw = Array.isArray(data?.latest_trades) ? data.latest_trades : [];
  const trades = tradesRaw
    .filter(trade => {
      if (!trade || typeof trade !== 'object') {
        return false;
      }
      const highlights = [
        trade.ticker, trade.Ticker, trade.symbol,
        trade.company_clean, trade.description, trade.Description,
        trade.transaction, trade.Transaction,
      ];
      return highlights.some(value => !!toText(value));
    })
    .slice(0, 5);

  if (trades.length < 1) {
    setSectionState(section, false);
    return;
  }

  trades.forEach(trade => {
    const rowEl = createTradeRowElement(trade);
    if (rowEl) {
      list.appendChild(rowEl);
    }
  });

  setSectionState(section, list.children.length > 0);
}

async function refresh() {
  if (inflight) return;
  inflight = true;

  try {
    const res = await fetch('/data.json?ts=' + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    if (DEBUG) {
      console.log('Fetched data:', data);
      console.log('Committees:', data.committees);
      console.log('Donors:', data.top_donors);
      console.log('Industries:', data.top_industries);
      console.log('Trades:', data.latest_trades);
    }

    if (!data) {
      console.error('Data is null or undefined');
      inflight = false;
      return;
    }
    
    // Reset error counter on successful fetch
    consecutiveErrors = 0;
    
    // Create fingerprint of important data to detect real changes
    const dataFingerprint = JSON.stringify({
      name: data.name,
      committees: (data.committees || []).length,
      donors: (data.top_donors || []).length,
      industries: (data.top_industries || []).length,
      trades: (data.latest_trades || []).length,
      holdings: (data.top_holdings || []).length,
      sectors: (data.top_traded_sectors || []).length
    });
    
    lastDataFingerprint = dataFingerprint;
    
    // Detect new speaker and reset to first card
    const currentSpeaker = toText(data.name);
    const isNewSpeaker = currentSpeaker && currentSpeaker !== lastSpeakerName;
    if (isNewSpeaker) {
      if (DEBUG) console.log('New speaker detected:', currentSpeaker, '(was:', lastSpeakerName, ')');
      lastSpeakerName = currentSpeaker;
      // Reset rotation to first card only for new speaker
      stopRotationTimer();
      stopDelayTimer();
      if (cardRotation.animationFrame) {
        cancelAnimationFrame(cardRotation.animationFrame);
        cardRotation.animationFrame = null;
      }
      cardRotation.isScrolling = false;
      cardRotation.currentIndex = 0;
    }

    if (data && data.ticker && Array.isArray(data.ticker.sections) && data.ticker.sections.length > 0) {
      const hash = JSON.stringify(data.ticker);
      if (tickerState.active && tickerState.dataHash === hash) {
        inflight = false;
        return;
      }
      enterTickerMode(data.ticker, hash);
      inflight = false;
      return;
    }

    if (tickerState.active) {
      exitTickerMode();
    }
    renderSpeakerName(data);
    renderNetworth(data);
    renderBioSection(data);
    renderDonorSection(data);
    renderIndustrySection(data);
    renderHoldingsSection(data);
    renderSectorsSection(data);
    renderCommitteesSection(data);
    renderTradesSection(data);

    reconcileCardRotation();

  } catch(e) {
    console.error('Refresh error:', e);
    consecutiveErrors++;
    
    if (consecutiveErrors >= MAX_ERRORS) {
      // Show error state to user
      const speakerNameEl = document.getElementById('speaker-name');
      if (speakerNameEl) {
        speakerNameEl.textContent = 'Connection lost - retrying...';
      }
    }
  } finally {
    inflight = false;
  }
}

// Polling management
let refreshInterval = null;

function startPolling() {
  if (refreshInterval) return; // Already running
  refresh(); // Immediate first refresh
  refreshInterval = setInterval(refresh, 5000);
}

function stopPolling() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

// Cleanup on page unload
window.addEventListener('beforeunload', stopPolling);

// Start polling
startPolling();
