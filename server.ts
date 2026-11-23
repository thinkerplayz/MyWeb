import 'dotenv/config';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express, { type NextFunction, type Request, type Response } from 'express';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import {
  INITIAL_QA_QUESTIONS,
  INITIAL_REVIEWS,
  INITIAL_SITE_CONFIG,
} from './src/data/projects.ts';
import type {
  AdminSecurityInfo,
  AdminStatus,
  AnalyticsEvent,
  ContactSubmission,
  InquiryCategory,
  Project,
  QaAnswer,
  QaCategory,
  QaCommentReply,
  QaQuestion,
  Review,
  SecurityTelemetry,
  SiteConfig,
} from './src/types/portfolio.ts';
import {
  buildOtpAuthUri,
  consumeBackupCode,
  generateBackupCodes,
  generateTotpSecret,
  getTotpSecondsRemaining,
  hashBackupCode,
  verifyTotpCode,
} from './src/utils/adminSecurity.ts';
import {
  aggregateAnalyticsSummary,
  checkSpamSubmission,
  type CooldownEntry,
  formatCooldownDuration,
  isValidEmailAddress,
  LoginCooldownTracker,
  maskEmailAddress,
  mergeSiteConfig,
  sanitizeExternalUrl,
  sanitizeSocialLinks,
  SlidingWindowRateLimiter,
  validateAnswerForm,
  validateContactForm,
  validateQuestionForm,
  validateReviewForm,
  validateStrongPassword,
} from './src/utils/portfolioLogic.ts';

interface AdminCredentialRecord {
  username: string;
  email: string;
  salt: string;
  passwordHash: string;
  totpSecret: string;
  backupCodeHashes: string[];
  cooldownHours: number;
  maxFailedAttempts: number;
  createdAt: string;
  updatedAt?: string;
}

interface PortfolioStore {
  reviews: Review[];
  contacts: ContactSubmission[];
  events: AnalyticsEvent[];
  questions: QaQuestion[];
  siteConfig: SiteConfig;
  adminCredential: AdminCredentialRecord | null;
  loginLockouts: Record<string, CooldownEntry>;
  sessionVersion: number;
  security: SecurityTelemetry;
}

const PORT = Number(process.env.PORT || 3000);
const DATA_DIR = path.resolve(process.cwd(), 'data');
const STORE_PATH = path.resolve(DATA_DIR, 'portfolio-store.json');
const SESSION_SECRET = process.env.ADMIN_SESSION_SECRET || crypto.randomBytes(32).toString('hex');
const TOKEN_TTL_MS = 12 * 60 * 60 * 1000;
const DEFAULT_COOLDOWN_HOURS = 4;
const DEFAULT_MAX_FAILED_ATTEMPTS = 5;

const ALLOWED_QA_CATEGORIES = new Set<QaCategory>([
  'Server Setup',
  'Onboarding',
  'Permissions & Roles',
  'Bots & Automation',
  'Monetization',
  'Creator Strategy',
]);

const ALLOWED_INQUIRY_CATEGORIES = new Set<InquiryCategory>([
  'project',
  'purchase',
  'support',
  'general',
]);

const writeRateLimiter = new SlidingWindowRateLimiter(15, 60_000);
const reactionRateLimiter = new SlidingWindowRateLimiter(50, 60_000);
const authRateLimiter = new SlidingWindowRateLimiter(12, 60_000);
const aiRateLimiter = new SlidingWindowRateLimiter(10, 60_000);
const loginCooldownTracker = new LoginCooldownTracker({
  freeAttempts: DEFAULT_MAX_FAILED_ATTEMPTS,
  baseCooldownMs: DEFAULT_COOLDOWN_HOURS * 60 * 60 * 1000,
  maxCooldownMs: 24 * 60 * 60 * 1000,
});
const revokedTokenIds = new Set<string>();
const recentSpamFingerprints = new Map<string, number>();

function getClientIp(req: Request): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string' && forwarded.trim()) {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket.remoteAddress || 'unknown';
}

function cloneDefaultSiteConfig(): SiteConfig {
  return JSON.parse(JSON.stringify(INITIAL_SITE_CONFIG)) as SiteConfig;
}

function createSeedEvents(): AnalyticsEvent[] {
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  const isoDay = (daysAgo: number, hourOffset: number) =>
    new Date(now - daysAgo * dayMs + hourOffset * 3600 * 1000).toISOString();

  return [
    { id: 'seed-e1', type: 'page_view', sessionId: 'vis-101', target: 'overview', timestamp: isoDay(4, 1) },
    { id: 'seed-e2', type: 'tab_view', sessionId: 'vis-101', target: 'gallery', timestamp: isoDay(4, 2) },
    { id: 'seed-e3', type: 'page_view', sessionId: 'vis-102', target: 'overview', timestamp: isoDay(3, 3) },
    { id: 'seed-e4', type: 'tab_view', sessionId: 'vis-102', target: 'community', timestamp: isoDay(3, 4) },
    { id: 'seed-e5', type: 'tab_view', sessionId: 'vis-103', target: 'about', timestamp: isoDay(2, 1) },
    { id: 'seed-e6', type: 'tab_view', sessionId: 'vis-103', target: 'contact', timestamp: isoDay(2, 2) },
    { id: 'seed-e7', type: 'page_view', sessionId: 'vis-104', target: 'overview', timestamp: isoDay(1, 1) },
    { id: 'seed-e8', type: 'tab_view', sessionId: 'vis-104', target: 'reviews', timestamp: isoDay(1, 2) },
    { id: 'seed-e9', type: 'page_view', sessionId: 'vis-105', target: 'overview', timestamp: isoDay(0, -2) },
  ];
}

function defaultSecurityTelemetry(): SecurityTelemetry {
  return {
    failedLoginAttempts: 0,
    cooldownTriggers: 0,
    rateLimitHits: 0,
    spamBlocked: 0,
  };
}

function syncCooldownConfig(store: PortfolioStore): void {
  const hours = store.adminCredential?.cooldownHours ?? DEFAULT_COOLDOWN_HOURS;
  const maxAttempts = store.adminCredential?.maxFailedAttempts ?? DEFAULT_MAX_FAILED_ATTEMPTS;
  loginCooldownTracker.configure({
    freeAttempts: maxAttempts,
    baseCooldownMs: hours * 60 * 60 * 1000,
    maxCooldownMs: Math.max(hours, 24) * 60 * 60 * 1000,
  });
  loginCooldownTracker.importState(store.loginLockouts);
}

function ensureStore(): PortfolioStore {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  if (!fs.existsSync(STORE_PATH)) {
    const initial: PortfolioStore = {
      reviews: [...INITIAL_REVIEWS],
      contacts: [],
      events: createSeedEvents(),
      questions: JSON.parse(JSON.stringify(INITIAL_QA_QUESTIONS)) as QaQuestion[],
      siteConfig: cloneDefaultSiteConfig(),
      adminCredential: null,
      loginLockouts: {},
      sessionVersion: 1,
      security: defaultSecurityTelemetry(),
    };
    fs.writeFileSync(STORE_PATH, JSON.stringify(initial, null, 2), 'utf8');
    syncCooldownConfig(initial);
    return initial;
  }
  try {
    const raw = fs.readFileSync(STORE_PATH, 'utf8');
    const parsed = JSON.parse(raw) as Partial<PortfolioStore>;
    let cred = parsed.adminCredential ?? null;
    if (cred) {
      cred = {
        ...cred,
        email: cred.email || '',
        totpSecret: cred.totpSecret || generateTotpSecret(),
        backupCodeHashes: Array.isArray(cred.backupCodeHashes) ? cred.backupCodeHashes : [],
        cooldownHours:
          typeof cred.cooldownHours === 'number'
            ? Math.min(24, Math.max(1, cred.cooldownHours))
            : DEFAULT_COOLDOWN_HOURS,
        maxFailedAttempts:
          typeof cred.maxFailedAttempts === 'number'
            ? Math.min(10, Math.max(3, cred.maxFailedAttempts))
            : DEFAULT_MAX_FAILED_ATTEMPTS,
      };
    }
    const store: PortfolioStore = {
      reviews: Array.isArray(parsed.reviews) ? parsed.reviews : [...INITIAL_REVIEWS],
      contacts: Array.isArray(parsed.contacts) ? parsed.contacts : [],
      events: Array.isArray(parsed.events) ? parsed.events : createSeedEvents(),
      questions: Array.isArray(parsed.questions)
        ? parsed.questions
        : (JSON.parse(JSON.stringify(INITIAL_QA_QUESTIONS)) as QaQuestion[]),
      siteConfig: mergeSiteConfig(INITIAL_SITE_CONFIG, parsed.siteConfig),
      adminCredential: cred,
      loginLockouts: parsed.loginLockouts && typeof parsed.loginLockouts === 'object' ? parsed.loginLockouts : {},
      sessionVersion: typeof parsed.sessionVersion === 'number' ? parsed.sessionVersion : 1,
      security: {
        ...defaultSecurityTelemetry(),
        ...(parsed.security ?? {}),
      },
    };
    syncCooldownConfig(store);
    return store;
  } catch {
    const fallback: PortfolioStore = {
      reviews: [...INITIAL_REVIEWS],
      contacts: [],
      events: createSeedEvents(),
      questions: JSON.parse(JSON.stringify(INITIAL_QA_QUESTIONS)) as QaQuestion[],
      siteConfig: cloneDefaultSiteConfig(),
      adminCredential: null,
      loginLockouts: {},
      sessionVersion: 1,
      security: defaultSecurityTelemetry(),
    };
    fs.writeFileSync(STORE_PATH, JSON.stringify(fallback, null, 2), 'utf8');
    syncCooldownConfig(fallback);
    return fallback;
  }
}

function saveStore(store: PortfolioStore): void {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  const tmpPath = `${STORE_PATH}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(store, null, 2), 'utf8');
  fs.renameSync(tmpPath, STORE_PATH);
}

function sanitizeAvatar(raw: unknown): string | undefined {
  const val = String(raw ?? '').trim();
  if (!val || val.length > 350000) return undefined;
  if (val.startsWith('/assets/') || val.startsWith('/src/assets/') || val.startsWith('data:image/')) {
    return val;
  }
  return undefined;
}

function hashPassword(password: string, salt: string): string {
  return crypto.scryptSync(password, salt, 64).toString('hex');
}

function safeCompareString(a: string, b: string): boolean {
  const bufA = Buffer.from(a, 'utf8');
  const bufB = Buffer.from(b, 'utf8');
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

interface VerifiedAdminPayload {
  sub: string;
  exp: number;
  jti: string;
  ver: number;
}

function signAdminToken(username: string, sessionVersion: number): string {
  const payload = Buffer.from(
    JSON.stringify({
      sub: username,
      role: 'admin',
      jti: crypto.randomUUID(),
      ver: sessionVersion,
      exp: Date.now() + TOKEN_TTL_MS,
    }),
    'utf8'
  ).toString('base64url');
  const signature = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(payload)
    .digest('base64url');
  return `${payload}.${signature}`;
}

function verifyAdminToken(token: string, expectedVersion: number): VerifiedAdminPayload | null {
  const parts = token.split('.');
  if (parts.length !== 2) return null;
  const [payload, signature] = parts;
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(payload)
    .digest('base64url');
  if (!safeCompareString(signature, expectedSig)) return null;
  try {
    const decoded = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as {
      sub: string;
      role: string;
      exp: number;
      jti?: string;
      ver?: number;
    };
    if (
      decoded.role !== 'admin' ||
      typeof decoded.exp !== 'number' ||
      decoded.exp < Date.now() ||
      !decoded.jti ||
      revokedTokenIds.has(decoded.jti) ||
      decoded.ver !== expectedVersion
    ) {
      return null;
    }
    return { sub: decoded.sub, exp: decoded.exp, jti: decoded.jti, ver: decoded.ver };
  } catch {
    return null;
  }
}

function getAdminStatus(store: PortfolioStore): AdminStatus {
  const envUser = (process.env.ADMIN_USERNAME || '').trim();
  const envPass = (process.env.ADMIN_PASSWORD || '').trim();
  if (envUser && envPass) {
    return {
      isConfigured: true,
      source: 'env',
      usernameHint: envUser,
      totpEnabled: Boolean((process.env.ADMIN_TOTP_SECRET || '').trim()),
      cooldownHours: DEFAULT_COOLDOWN_HOURS,
      maxFailedAttempts: DEFAULT_MAX_FAILED_ATTEMPTS,
    };
  }
  if (store.adminCredential) {
    return {
      isConfigured: true,
      source: 'claimed',
      usernameHint: store.adminCredential.username,
      emailMasked: store.adminCredential.email ? maskEmailAddress(store.adminCredential.email) : undefined,
      totpEnabled: Boolean(store.adminCredential.totpSecret),
      backupCodesRemaining: store.adminCredential.backupCodeHashes.length,
      cooldownHours: store.adminCredential.cooldownHours,
      maxFailedAttempts: store.adminCredential.maxFailedAttempts,
    };
  }
  return {
    isConfigured: false,
    source: 'unclaimed',
    cooldownHours: DEFAULT_COOLDOWN_HOURS,
    maxFailedAttempts: DEFAULT_MAX_FAILED_ATTEMPTS,
  };
}

function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  const authHeader = req.headers.authorization || '';
  if (!authHeader.startsWith('Bearer ')) {
    res.status(401).json({ error: 'Authentication required.' });
    return;
  }
  const token = authHeader.slice('Bearer '.length).trim();
  const store = ensureStore();
  const verified = verifyAdminToken(token, store.sessionVersion);
  if (!verified) {
    res.status(401).json({ error: 'Session expired or invalid credentials.' });
    return;
  }
  res.locals.adminAuth = verified;
  next();
}

function enforceRateLimit(
  limiter: SlidingWindowRateLimiter,
  scope: string,
  req: Request,
  res: Response
): boolean {
  const ip = getClientIp(req);
  const check = limiter.consume(`${scope}:${ip}`);
  if (!check.allowed) {
    const store = ensureStore();
    store.security.rateLimitHits += 1;
    saveStore(store);
    res.setHeader('Retry-After', String(check.retryAfterSeconds));
    res.status(429).json({
      error: `Too many requests. Please wait ${check.retryAfterSeconds}s and try again.`,
      retryAfterSeconds: check.retryAfterSeconds,
    });
    return false;
  }
  return true;
}

function sanitizeProjectPayload(raw: Project): Project {
  const id = String(raw.id || `proj-${crypto.randomUUID().slice(0, 8)}`).trim().slice(0, 80);
  const title = String(raw.title || 'Untitled Project').trim().slice(0, 120);
  const slug = String(raw.slug || id).trim().toLowerCase().replace(/[^a-z0-9-]/g, '-').slice(0, 80) || id;
  const validCategories = new Set([
    'Community Hubs',
    'Creator Guilds',
    'Automation & Security',
    'Monetization Systems',
  ]);
  const category = validCategories.has(raw.category) ? raw.category : 'Community Hubs';

  return {
    id,
    slug,
    title,
    client: String(raw.client || 'Independent Client').trim().slice(0, 100),
    category,
    year: String(raw.year || new Date().getFullYear()).trim().slice(0, 12),
    summary: String(raw.summary || '').trim().slice(0, 400),
    overview: String(raw.overview || '').trim().slice(0, 1600),
    challenge: String(raw.challenge || '').trim().slice(0, 1200),
    solution: String(raw.solution || '').trim().slice(0, 1200),
    imageUrl: sanitizeExternalUrl(raw.imageUrl, 'image') || '',
    symbolUrl: sanitizeExternalUrl(raw.symbolUrl, 'image') || '/assets/aistudio/60330-developer_bb7c9d1d.png',
    featured: Boolean(raw.featured),
    deliverables: Array.isArray(raw.deliverables)
      ? raw.deliverables.map((d) => String(d).trim().slice(0, 120)).filter(Boolean).slice(0, 12)
      : [],
    techStack: Array.isArray(raw.techStack)
      ? raw.techStack.map((t) => String(t).trim().slice(0, 60)).filter(Boolean).slice(0, 12)
      : [],
    rolesDesigned: Array.isArray(raw.rolesDesigned)
      ? raw.rolesDesigned.slice(0, 10).map((r) => ({
          name: String(r?.name || 'Role').trim().slice(0, 60),
          tier: String(r?.tier || 'Tier').trim().slice(0, 40),
          permissionsSummary: String(r?.permissionsSummary || '').trim().slice(0, 240),
        }))
      : [],
    channelBlueprint: Array.isArray(raw.channelBlueprint)
      ? raw.channelBlueprint.slice(0, 8).map((g) => ({
          categoryName: String(g?.categoryName || 'Category').trim().slice(0, 60),
          accessLevel: String(g?.accessLevel || 'Members').trim().slice(0, 60),
          channels: Array.isArray(g?.channels)
            ? g.channels.slice(0, 10).map((ch) => ({
                name: String(ch?.name || 'general').trim().slice(0, 60),
                kind: ['text', 'voice', 'forum', 'stage', 'announcement'].includes(ch?.kind)
                  ? ch.kind
                  : 'text',
                purpose: String(ch?.purpose || '').trim().slice(0, 160),
              }))
            : [],
        }))
      : [],
    metrics: Array.isArray(raw.metrics)
      ? raw.metrics.slice(0, 6).map((m) => ({
          label: String(m?.label || 'Metric').trim().slice(0, 50),
          value: String(m?.value || '0').trim().slice(0, 30),
          context: String(m?.context || '').trim().slice(0, 120),
        }))
      : [],
  };
}

function getAiClient(): GoogleGenAI | null {
  const apiKey = (process.env.GEMINI_API_KEY || '').trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer(): Promise<void> {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', 1);

  // Iframe-friendly baseline security headers (no X-Frame-Options so AI Studio preview works)
  app.use((_req: Request, res: Response, next: NextFunction) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
    next();
  });

  app.use(express.json({ limit: '25mb' }));
  app.use(express.urlencoded({ extended: true, limit: '25mb' }));

  // Public Site Config API
  app.get('/api/site-config', (_req: Request, res: Response) => {
    const store = ensureStore();
    res.json({ siteConfig: store.siteConfig });
  });

  // Public Reviews API
  app.get('/api/reviews', (_req: Request, res: Response) => {
    const store = ensureStore();
    res.json({ reviews: store.reviews });
  });

  app.post('/api/reviews', (req: Request, res: Response) => {
    if (!enforceRateLimit(writeRateLimiter, 'reviews', req, res)) return;

    const body = req.body ?? {};
    const authorName = String(body.authorName ?? '').trim().slice(0, 80);
    const authorRole = String(body.authorRole ?? 'Community Member').trim().slice(0, 80) || 'Community Member';
    const rating = Number(body.rating);
    const targetProjectId = String(body.targetProjectId ?? 'general').trim();
    const comment = String(body.comment ?? '').trim().slice(0, 800);
    const honeypot = String(body.websiteUrl ?? '');

    const spam = checkSpamSubmission({
      honeypot,
      text: `${authorName} ${comment}`,
      recentFingerprints: recentSpamFingerprints,
      fingerprintKey: `review:${getClientIp(req)}`,
    });
    if (spam.isSpam) {
      const store = ensureStore();
      store.security.spamBlocked += 1;
      saveStore(store);
      res.status(400).json({ error: spam.reason || 'Submission flagged by anti-spam protection.' });
      return;
    }

    const validation = validateReviewForm({ authorName, authorRole, rating, comment });
    if (!validation.valid) {
      res.status(400).json({ error: 'Invalid review input.', errors: validation.errors });
      return;
    }

    const store = ensureStore();
    const matchedProject = store.siteConfig.projects.find((p) => p.id === targetProjectId);
    const targetProjectTitle = matchedProject
      ? matchedProject.title
      : 'Portfolio & Creator Guide Experience';

    const newReview: Review = {
      id: `rev-${crypto.randomUUID().slice(0, 8)}`,
      authorName,
      authorRole,
      authorAvatar: sanitizeAvatar(body.authorAvatar),
      rating: Math.min(5, Math.max(1, Math.round(rating))),
      targetProjectId: matchedProject ? matchedProject.id : 'general',
      targetProjectTitle,
      comment,
      createdAt: new Date().toISOString(),
    };

    store.reviews.unshift(newReview);
    store.events.push({
      id: `ev-${crypto.randomUUID().slice(0, 8)}`,
      type: 'review_submitted',
      sessionId: String(body.sessionId || 'anon').slice(0, 64),
      target: newReview.targetProjectId,
      timestamp: new Date().toISOString(),
    });
    saveStore(store);

    res.status(201).json({ review: newReview });
  });

  // Public Contact & Purchase Inquiry API
  app.post('/api/contact', (req: Request, res: Response) => {
    if (!enforceRateLimit(writeRateLimiter, 'contact', req, res)) return;

    const body = req.body ?? {};
    const name = String(body.name ?? '').trim().slice(0, 80);
    const contactHandle = String(body.contactHandle ?? '').trim().slice(0, 120);
    const rawCat = String(body.category ?? 'project').trim() as InquiryCategory;
    const category: InquiryCategory = ALLOWED_INQUIRY_CATEGORIES.has(rawCat) ? rawCat : 'project';
    const serviceScope = String(body.serviceScope ?? '').trim().slice(0, 100);
    const budgetRange = String(body.budgetRange ?? 'Flexible').trim().slice(0, 60);
    const serverSize = String(body.serverSize ?? '').trim().slice(0, 60) || undefined;
    const timeline = String(body.timeline ?? '').trim().slice(0, 60) || undefined;
    const preferredChannel = String(body.preferredChannel ?? '').trim().slice(0, 40) || undefined;
    const existingServerUrl = String(body.existingServerUrl ?? '').trim().slice(0, 160) || undefined;
    const requestedFeatures = Array.isArray(body.requestedFeatures)
      ? body.requestedFeatures.map((f: unknown) => String(f).trim().slice(0, 50)).filter(Boolean).slice(0, 8)
      : undefined;
    const message = String(body.message ?? '').trim().slice(0, 2000);
    const honeypot = String(body.websiteUrl ?? '');

    const spam = checkSpamSubmission({
      honeypot,
      text: `${name} ${message}`,
      recentFingerprints: recentSpamFingerprints,
      fingerprintKey: `contact:${getClientIp(req)}`,
    });
    if (spam.isSpam) {
      const store = ensureStore();
      store.security.spamBlocked += 1;
      saveStore(store);
      res.status(400).json({ error: spam.reason || 'Submission blocked by anti-spam protection.' });
      return;
    }

    const validation = validateContactForm({
      name,
      contactHandle,
      serviceScope,
      budgetRange,
      existingServerUrl,
      message,
    });
    if (!validation.valid) {
      res.status(400).json({ error: 'Please complete all required fields.', errors: validation.errors });
      return;
    }

    const submission: ContactSubmission = {
      id: `msg-${crypto.randomUUID().slice(0, 8)}`,
      name,
      contactHandle,
      category,
      serviceScope,
      budgetRange,
      serverSize,
      timeline,
      preferredChannel,
      existingServerUrl,
      requestedFeatures,
      message,
      status: 'unread',
      createdAt: new Date().toISOString(),
    };

    const store = ensureStore();
    store.contacts.unshift(submission);
    store.events.push({
      id: `ev-${crypto.randomUUID().slice(0, 8)}`,
      type: 'contact_submitted',
      sessionId: String(body.sessionId || 'anon').slice(0, 64),
      target: `${category}:${serviceScope}`,
      timestamp: new Date().toISOString(),
    });
    saveStore(store);

    res.status(201).json({
      id: submission.id,
      createdAt: submission.createdAt,
      message: 'Inquiry received by Thinker.',
    });
  });

  // Public Community Q&A Forum API
  app.get('/api/questions', (_req: Request, res: Response) => {
    const store = ensureStore();
    res.json({ questions: store.questions });
  });

  app.post('/api/questions', (req: Request, res: Response) => {
    if (!enforceRateLimit(writeRateLimiter, 'qa-question', req, res)) return;

    const body = req.body ?? {};
    const authorName = String(body.authorName ?? '').trim().slice(0, 60);
    const title = String(body.title ?? '').trim().slice(0, 180);
    const qBody = String(body.body ?? '').trim().slice(0, 3000);
    const honeypot = String(body.websiteUrl ?? '');

    const spam = checkSpamSubmission({
      honeypot,
      text: `${title} ${qBody}`,
      recentFingerprints: recentSpamFingerprints,
      fingerprintKey: `question:${getClientIp(req)}`,
    });
    if (spam.isSpam) {
      const store = ensureStore();
      store.security.spamBlocked += 1;
      saveStore(store);
      res.status(400).json({ error: spam.reason || 'Question flagged by anti-spam protection.' });
      return;
    }

    const rawCat = String(body.category ?? 'Server Setup').trim() as QaCategory;
    const category: QaCategory = ALLOWED_QA_CATEGORIES.has(rawCat) ? rawCat : 'Server Setup';
    const sessionId = String(body.sessionId ?? 'anon').trim().slice(0, 64) || 'anon';
    const authorBio = String(body.authorBio ?? '').trim().slice(0, 140);
    const rawTags: string[] = Array.isArray(body.tags)
      ? body.tags.map((t: unknown) => String(t))
      : String(body.tags ?? '').split(',');
    const tags = Array.from(
      new Set(
        rawTags
          .map((t) => t.trim().toLowerCase().replace(/^#+/, '').replace(/[^a-z0-9-]/g, '-'))
          .filter((t) => t.length >= 2 && t.length <= 24)
      )
    ).slice(0, 5);

    const check = validateQuestionForm({ authorName, title, body: qBody });
    if (!check.valid) {
      res.status(400).json({ error: 'Please complete the question form.', errors: check.errors });
      return;
    }

    const newQuestion: QaQuestion = {
      id: `qa-${crypto.randomUUID().slice(0, 8)}`,
      title,
      body: qBody,
      category,
      tags: tags.length > 0 ? tags : [category.toLowerCase().replace(/[^a-z0-9]+/g, '-')],
      authorName,
      authorAvatar: sanitizeAvatar(body.authorAvatar),
      authorBio: authorBio || undefined,
      authorSocials: sanitizeSocialLinks(body.authorSocials),
      likes: 1,
      likedBySessions: [sessionId],
      followerSessions: [sessionId],
      answers: [],
      views: 1,
      isPinned: false,
      createdAt: new Date().toISOString(),
    };

    const store = ensureStore();
    store.questions.unshift(newQuestion);
    store.events.push({
      id: `ev-${crypto.randomUUID().slice(0, 8)}`,
      type: 'qa_activity',
      sessionId,
      target: `question:${newQuestion.id}`,
      timestamp: new Date().toISOString(),
    });
    saveStore(store);

    res.status(201).json({ question: newQuestion });
  });

  app.post('/api/questions/:id/view', (req: Request, res: Response) => {
    if (!enforceRateLimit(reactionRateLimiter, 'qa-view', req, res)) return;
    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    q.views += 1;
    saveStore(store);
    res.json({ views: q.views });
  });

  app.post('/api/questions/:id/like', (req: Request, res: Response) => {
    if (!enforceRateLimit(reactionRateLimiter, 'qa-like', req, res)) return;
    const sessionId = String(req.body?.sessionId ?? 'anon').trim().slice(0, 64) || 'anon';
    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    const exists = q.likedBySessions.includes(sessionId);
    if (exists) {
      q.likedBySessions = q.likedBySessions.filter((id) => id !== sessionId);
      q.likes = Math.max(0, q.likes - 1);
    } else {
      q.likedBySessions.push(sessionId);
      q.likes += 1;
    }
    saveStore(store);
    res.json({ question: q });
  });

  app.post('/api/questions/:id/follow', (req: Request, res: Response) => {
    if (!enforceRateLimit(reactionRateLimiter, 'qa-follow', req, res)) return;
    const sessionId = String(req.body?.sessionId ?? 'anon').trim().slice(0, 64) || 'anon';
    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    const exists = q.followerSessions.includes(sessionId);
    if (exists) {
      q.followerSessions = q.followerSessions.filter((id) => id !== sessionId);
    } else {
      q.followerSessions.push(sessionId);
    }
    saveStore(store);
    res.json({ question: q });
  });

  app.post('/api/questions/:id/answers', (req: Request, res: Response) => {
    if (!enforceRateLimit(writeRateLimiter, 'qa-answer', req, res)) return;
    const body = req.body ?? {};
    const authorName = String(body.authorName ?? '').trim().slice(0, 60);
    const ansBody = String(body.body ?? '').trim().slice(0, 3000);
    const sessionId = String(body.sessionId ?? 'anon').trim().slice(0, 64) || 'anon';

    const spam = checkSpamSubmission({
      honeypot: String(body.websiteUrl ?? ''),
      text: ansBody,
      recentFingerprints: recentSpamFingerprints,
      fingerprintKey: `answer:${getClientIp(req)}`,
    });
    if (spam.isSpam) {
      const store = ensureStore();
      store.security.spamBlocked += 1;
      saveStore(store);
      res.status(400).json({ error: spam.reason || 'Answer flagged by anti-spam protection.' });
      return;
    }

    const check = validateAnswerForm({ authorName, body: ansBody });
    if (!check.valid) {
      res.status(400).json({ error: 'Invalid answer input.', errors: check.errors });
      return;
    }

    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }

    const newAnswer: QaAnswer = {
      id: `ans-${crypto.randomUUID().slice(0, 8)}`,
      authorName,
      authorAvatar: sanitizeAvatar(body.authorAvatar),
      authorSocials: sanitizeSocialLinks(body.authorSocials),
      body: ansBody,
      likes: 0,
      likedBySessions: [],
      isAccepted: false,
      replies: [],
      createdAt: new Date().toISOString(),
    };

    q.answers.push(newAnswer);
    store.events.push({
      id: `ev-${crypto.randomUUID().slice(0, 8)}`,
      type: 'qa_activity',
      sessionId,
      target: `answer:${q.id}`,
      timestamp: new Date().toISOString(),
    });
    saveStore(store);

    res.status(201).json({ question: q, answer: newAnswer });
  });

  app.post('/api/questions/:id/answers/:answerId/like', (req: Request, res: Response) => {
    if (!enforceRateLimit(reactionRateLimiter, 'qa-answer-like', req, res)) return;
    const sessionId = String(req.body?.sessionId ?? 'anon').trim().slice(0, 64) || 'anon';
    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    const answer = q.answers.find((a) => a.id === req.params.answerId);
    if (!answer) {
      res.status(404).json({ error: 'Answer not found.' });
      return;
    }
    const exists = answer.likedBySessions.includes(sessionId);
    if (exists) {
      answer.likedBySessions = answer.likedBySessions.filter((id) => id !== sessionId);
      answer.likes = Math.max(0, answer.likes - 1);
    } else {
      answer.likedBySessions.push(sessionId);
      answer.likes += 1;
    }
    saveStore(store);
    res.json({ question: q });
  });

  app.post('/api/questions/:id/answers/:answerId/replies', (req: Request, res: Response) => {
    if (!enforceRateLimit(writeRateLimiter, 'qa-reply', req, res)) return;
    const body = req.body ?? {};
    const authorName = String(body.authorName ?? '').trim().slice(0, 60) || 'Guest Builder';
    const replyBody = String(body.body ?? '').trim().slice(0, 1200);
    if (replyBody.length < 2) {
      res.status(400).json({ error: 'Reply cannot be empty.' });
      return;
    }

    const spam = checkSpamSubmission({
      honeypot: String(body.websiteUrl ?? ''),
      text: replyBody,
      recentFingerprints: recentSpamFingerprints,
      fingerprintKey: `reply:${getClientIp(req)}`,
    });
    if (spam.isSpam) {
      const store = ensureStore();
      store.security.spamBlocked += 1;
      saveStore(store);
      res.status(400).json({ error: spam.reason || 'Reply flagged by anti-spam protection.' });
      return;
    }

    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    const answer = q.answers.find((a) => a.id === req.params.answerId);
    if (!answer) {
      res.status(404).json({ error: 'Answer not found.' });
      return;
    }

    const reply: QaCommentReply = {
      id: `rep-${crypto.randomUUID().slice(0, 8)}`,
      authorName,
      authorAvatar: sanitizeAvatar(body.authorAvatar),
      authorSocials: sanitizeSocialLinks(body.authorSocials),
      body: replyBody,
      createdAt: new Date().toISOString(),
    };

    answer.replies.push(reply);
    saveStore(store);
    res.status(201).json({ question: q, reply });
  });

  // Public Analytics Event Ingestion
  app.post('/api/analytics/event', (req: Request, res: Response) => {
    if (!enforceRateLimit(reactionRateLimiter, 'analytics', req, res)) return;
    const body = req.body ?? {};
    const allowedTypes = new Set([
      'page_view',
      'tab_view',
      'project_view',
      'guide_query',
      'qa_activity',
    ]);
    const type = String(body.type ?? '');
    if (!allowedTypes.has(type)) {
      res.status(400).json({ error: 'Unsupported analytics event type.' });
      return;
    }
    const sessionId = String(body.sessionId ?? 'anon').trim().slice(0, 64);
    const target = String(body.target ?? 'overview').trim().slice(0, 100);

    const store = ensureStore();
    store.events.push({
      id: `ev-${crypto.randomUUID().slice(0, 8)}`,
      type: type as AnalyticsEvent['type'],
      sessionId,
      target,
      timestamp: new Date().toISOString(),
    });
    if (store.events.length > 2000) {
      store.events = store.events.slice(-1500);
    }
    saveStore(store);
    res.status(202).json({ ok: true });
  });

  // Admin Auth Routes (Connected Email + 30s TOTP 2FA + Backup Codes + Multi-Hour Cooldowns + Rate Limits)
  app.get('/api/admin/status', (_req: Request, res: Response) => {
    const store = ensureStore();
    res.json(getAdminStatus(store));
  });

  app.post('/api/admin/bootstrap', (req: Request, res: Response) => {
    if (!enforceRateLimit(authRateLimiter, 'admin-bootstrap', req, res)) return;
    const store = ensureStore();
    const status = getAdminStatus(store);
    if (status.isConfigured) {
      res.status(409).json({ error: 'Admin portal is already configured. Please sign in.' });
      return;
    }
    const username = String(req.body?.username ?? '').trim();
    const email = String(req.body?.email ?? '').trim().toLowerCase();
    const password = String(req.body?.password ?? '');
    const cooldownHoursRaw = Number(req.body?.cooldownHours ?? DEFAULT_COOLDOWN_HOURS);
    const cooldownHours = Number.isFinite(cooldownHoursRaw)
      ? Math.min(24, Math.max(1, Math.round(cooldownHoursRaw)))
      : DEFAULT_COOLDOWN_HOURS;

    if (username.length < 3 || username.length > 48 || !/^[a-zA-Z0-9._-]+$/.test(username)) {
      res.status(400).json({
        error: 'Username must be 3–48 characters (letters, numbers, dot, underscore, or hyphen).',
      });
      return;
    }
    if (!isValidEmailAddress(email)) {
      res.status(400).json({
        error: 'Please enter a valid admin email address for security alerts & verification.',
      });
      return;
    }
    const passCheck = validateStrongPassword(password);
    if (!passCheck.valid) {
      res.status(400).json({ error: passCheck.reason || 'Password does not meet security requirements.' });
      return;
    }

    const salt = crypto.randomBytes(16).toString('hex');
    const totpSecret = generateTotpSecret();
    const backupCodes = generateBackupCodes(8);
    const backupCodeHashes = backupCodes.map((code) => hashBackupCode(code, salt));

    store.adminCredential = {
      username,
      email,
      salt,
      passwordHash: hashPassword(password, salt),
      totpSecret,
      backupCodeHashes,
      cooldownHours,
      maxFailedAttempts: DEFAULT_MAX_FAILED_ATTEMPTS,
      createdAt: new Date().toISOString(),
    };
    store.sessionVersion += 1;
    store.loginLockouts = {};
    store.security.lastAdminLoginAt = new Date().toISOString();
    syncCooldownConfig(store);
    saveStore(store);

    const otpauthUri = buildOtpAuthUri({
      issuer: 'Thinker Admin',
      accountName: email,
      secret: totpSecret,
    });
    const token = signAdminToken(username, store.sessionVersion);
    res.status(201).json({
      token,
      username,
      email,
      cooldownHours,
      totpSetup: {
        secret: totpSecret,
        otpauthUri,
        issuer: 'Thinker Admin',
        accountName: email,
      },
      backupCodes,
    });
  });

  app.post('/api/admin/login', (req: Request, res: Response) => {
    if (!enforceRateLimit(authRateLimiter, 'admin-login', req, res)) return;

    const store = ensureStore();
    const ip = getClientIp(req);
    const username = String(req.body?.username ?? '').trim();
    const password = String(req.body?.password ?? '');
    const totpCode = String(req.body?.totpCode ?? '').trim();
    const backupCode = String(req.body?.backupCode ?? '').trim();
    const emailInput = String(req.body?.email ?? '').trim().toLowerCase();

    const cooldownKey = `${ip}:${username.toLowerCase() || 'any'}`;
    const ipCooldownKey = `ip:${ip}`;

    const activeCheck = loginCooldownTracker.check(cooldownKey);
    const ipCheck = loginCooldownTracker.check(ipCooldownKey);
    if (!activeCheck.allowed || !ipCheck.allowed) {
      const retryAfterSeconds = Math.max(activeCheck.retryAfterSeconds, ipCheck.retryAfterSeconds);
      const cooldownUntil = Math.max(activeCheck.cooldownUntil, ipCheck.cooldownUntil);
      res.setHeader('Retry-After', String(retryAfterSeconds));
      res.status(429).json({
        error: `Too many failed sign-in attempts. Multi-hour security cooldown active for ${formatCooldownDuration(retryAfterSeconds)}.`,
        retryAfterSeconds,
        cooldownUntil,
      });
      return;
    }

    const status = getAdminStatus(store);
    if (!status.isConfigured) {
      res.status(400).json({ error: 'Admin account not yet claimed. Complete first-run setup.' });
      return;
    }

    const recordBadAttempt = (customMessage?: string) => {
      const result = loginCooldownTracker.recordFailure(cooldownKey);
      const ipResult = loginCooldownTracker.recordFailure(ipCooldownKey);
      store.security.failedLoginAttempts += 1;
      if (result.cooldownTriggered || ipResult.cooldownTriggered) {
        store.security.cooldownTriggers += 1;
      }
      store.loginLockouts = loginCooldownTracker.exportState();
      saveStore(store);

      const retryAfterSeconds = Math.max(result.retryAfterSeconds, ipResult.retryAfterSeconds);
      const cooldownUntil = Math.max(result.cooldownUntil, ipResult.cooldownUntil);
      if (retryAfterSeconds > 0) {
        res.setHeader('Retry-After', String(retryAfterSeconds));
        res.status(429).json({
          error: `Too many failed sign-in attempts. Security lockout active for ${formatCooldownDuration(retryAfterSeconds)}.`,
          retryAfterSeconds,
          cooldownUntil,
        });
        return;
      }
      const remaining = Math.min(result.attemptsRemaining, ipResult.attemptsRemaining);
      const baseMsg = customMessage || 'Invalid credentials or verification code.';
      res.status(401).json({
        error:
          remaining > 0
            ? `${baseMsg} ${remaining} ${remaining === 1 ? 'attempt' : 'attempts'} remaining before multi-hour lockout.`
            : baseMsg,
      });
    };

    if (status.source === 'env') {
      const envUser = (process.env.ADMIN_USERNAME || '').trim();
      const envPass = (process.env.ADMIN_PASSWORD || '').trim();
      const envTotpSecret = (process.env.ADMIN_TOTP_SECRET || '').trim();
      if (!safeCompareString(username, envUser) || !safeCompareString(password, envPass)) {
        recordBadAttempt('Invalid username or password.');
        return;
      }
      if (envTotpSecret) {
        if (!totpCode) {
          res.status(401).json({
            requiresTwoFactor: true,
            error: 'Enter your 6-digit 30-second verification code.',
          });
          return;
        }
        if (!verifyTotpCode(envTotpSecret, totpCode)) {
          recordBadAttempt('Invalid 30-second verification code.');
          return;
        }
      }
      loginCooldownTracker.recordSuccess(cooldownKey);
      loginCooldownTracker.recordSuccess(ipCooldownKey);
      store.loginLockouts = loginCooldownTracker.exportState();
      store.security.lastAdminLoginAt = new Date().toISOString();
      saveStore(store);
      const token = signAdminToken(envUser, store.sessionVersion);
      res.json({ token, username: envUser });
      return;
    }

    const cred = store.adminCredential;
    if (!cred) {
      recordBadAttempt('Invalid username or password.');
      return;
    }
    const computed = hashPassword(password, cred.salt);
    if (!safeCompareString(username, cred.username) || !safeCompareString(computed, cred.passwordHash)) {
      recordBadAttempt('Invalid username or password.');
      return;
    }

    // Optional connected-email verification check if supplied
    if (emailInput && cred.email && !safeCompareString(emailInput, cred.email)) {
      recordBadAttempt('Connected admin email does not match.');
      return;
    }

    // Require 30-second TOTP verification code or single-use backup code on EVERY admin login
    if (!totpCode && !backupCode) {
      res.status(401).json({
        requiresTwoFactor: true,
        emailMasked: cred.email ? maskEmailAddress(cred.email) : undefined,
        totpSecondsRemaining: getTotpSecondsRemaining(),
        error: 'Password verified. Enter your 6-digit 30s verification code or a single-use backup code.',
      });
      return;
    }

    let usedBackupCode = false;
    if (totpCode) {
      if (!verifyTotpCode(cred.totpSecret, totpCode, Date.now(), 1)) {
        recordBadAttempt('Invalid or expired 30-second verification code.');
        return;
      }
    } else if (backupCode) {
      const consumed = consumeBackupCode(backupCode, cred.backupCodeHashes, cred.salt);
      if (!consumed.matched) {
        recordBadAttempt('Invalid or already-used recovery backup code.');
        return;
      }
      cred.backupCodeHashes = consumed.remainingHashes;
      usedBackupCode = true;
    }

    loginCooldownTracker.recordSuccess(cooldownKey);
    loginCooldownTracker.recordSuccess(ipCooldownKey);
    store.loginLockouts = loginCooldownTracker.exportState();
    store.security.lastAdminLoginAt = new Date().toISOString();
    saveStore(store);
    const token = signAdminToken(cred.username, store.sessionVersion);
    res.json({
      token,
      username: cred.username,
      backupCodeUsed: usedBackupCode,
      backupCodesRemaining: cred.backupCodeHashes.length,
    });
  });

  app.post('/api/admin/logout', requireAdmin, (_req: Request, res: Response) => {
    const auth = res.locals.adminAuth as VerifiedAdminPayload | undefined;
    if (auth?.jti) {
      revokedTokenIds.add(auth.jti);
    }
    res.json({ ok: true });
  });

  app.post('/api/admin/change-password', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    const status = getAdminStatus(store);
    if (status.source !== 'claimed' || !store.adminCredential) {
      res.status(400).json({
        error: 'Credentials managed by environment variables cannot be changed from the dashboard.',
      });
      return;
    }
    const currentPassword = String(req.body?.currentPassword ?? '');
    const newPassword = String(req.body?.newPassword ?? '');
    const newUsername = String(req.body?.username ?? store.adminCredential.username).trim();

    if (newUsername.length < 3 || newUsername.length > 48 || !/^[a-zA-Z0-9._-]+$/.test(newUsername)) {
      res.status(400).json({ error: 'Username must be 3–48 alphanumeric/dot/underscore/hyphen chars.' });
      return;
    }

    const currentComputed = hashPassword(currentPassword, store.adminCredential.salt);
    if (!safeCompareString(currentComputed, store.adminCredential.passwordHash)) {
      res.status(401).json({ error: 'Current password is incorrect.' });
      return;
    }

    const strength = validateStrongPassword(newPassword);
    if (!strength.valid) {
      res.status(400).json({ error: strength.reason || 'New password is too weak.' });
      return;
    }

    const newSalt = crypto.randomBytes(16).toString('hex');
    const freshBackupCodes = generateBackupCodes(8);
    store.adminCredential = {
      ...store.adminCredential,
      username: newUsername,
      salt: newSalt,
      passwordHash: hashPassword(newPassword, newSalt),
      backupCodeHashes: freshBackupCodes.map((code) => hashBackupCode(code, newSalt)),
      updatedAt: new Date().toISOString(),
    };
    store.sessionVersion += 1;
    saveStore(store);

    const token = signAdminToken(newUsername, store.sessionVersion);
    res.json({
      token,
      username: newUsername,
      backupCodes: freshBackupCodes,
      message: 'Password updated, sessions rotated, and 8 fresh backup codes issued.',
    });
  });

  app.get('/api/admin/security', requireAdmin, (_req: Request, res: Response) => {
    const store = ensureStore();
    const cred = store.adminCredential;
    const securityInfo: AdminSecurityInfo = {
      email: cred?.email || '',
      totpEnabled: Boolean(cred?.totpSecret),
      backupCodesRemaining: cred?.backupCodeHashes.length ?? 0,
      cooldownHours: cred?.cooldownHours ?? DEFAULT_COOLDOWN_HOURS,
      maxFailedAttempts: cred?.maxFailedAttempts ?? DEFAULT_MAX_FAILED_ATTEMPTS,
      totpSecondsRemaining: getTotpSecondsRemaining(),
    };
    res.json({ securityInfo });
  });

  app.patch('/api/admin/security/settings', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    if (!store.adminCredential) {
      res.status(400).json({ error: 'Environment-managed admin security cannot be edited here.' });
      return;
    }

    if (req.body?.email !== undefined) {
      const nextEmail = String(req.body.email).trim().toLowerCase();
      if (!isValidEmailAddress(nextEmail)) {
        res.status(400).json({ error: 'Please provide a valid verification email address.' });
        return;
      }
      store.adminCredential.email = nextEmail;
    }

    if (req.body?.cooldownHours !== undefined) {
      const hrs = Number(req.body.cooldownHours);
      if (!Number.isFinite(hrs) || hrs < 1 || hrs > 24) {
        res.status(400).json({ error: 'Cooldown duration must be between 1 and 24 hours.' });
        return;
      }
      store.adminCredential.cooldownHours = Math.round(hrs);
    }

    if (req.body?.maxFailedAttempts !== undefined) {
      const attempts = Number(req.body.maxFailedAttempts);
      if (!Number.isFinite(attempts) || attempts < 3 || attempts > 10) {
        res.status(400).json({ error: 'Failed attempts threshold must be between 3 and 10.' });
        return;
      }
      store.adminCredential.maxFailedAttempts = Math.round(attempts);
    }

    store.adminCredential.updatedAt = new Date().toISOString();
    syncCooldownConfig(store);
    saveStore(store);

    res.json({
      securityInfo: {
        email: store.adminCredential.email,
        totpEnabled: Boolean(store.adminCredential.totpSecret),
        backupCodesRemaining: store.adminCredential.backupCodeHashes.length,
        cooldownHours: store.adminCredential.cooldownHours,
        maxFailedAttempts: store.adminCredential.maxFailedAttempts,
        totpSecondsRemaining: getTotpSecondsRemaining(),
      } satisfies AdminSecurityInfo,
    });
  });

  app.post('/api/admin/security/backup-codes', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    if (!store.adminCredential) {
      res.status(400).json({ error: 'Claimed admin credentials required.' });
      return;
    }
    const currentPassword = String(req.body?.currentPassword ?? '');
    const computed = hashPassword(currentPassword, store.adminCredential.salt);
    if (!safeCompareString(computed, store.adminCredential.passwordHash)) {
      res.status(401).json({ error: 'Current password is required to regenerate backup codes.' });
      return;
    }
    const backupCodes = generateBackupCodes(8);
    store.adminCredential.backupCodeHashes = backupCodes.map((code) =>
      hashBackupCode(code, store.adminCredential!.salt)
    );
    store.adminCredential.updatedAt = new Date().toISOString();
    saveStore(store);
    res.json({
      backupCodes,
      backupCodesRemaining: backupCodes.length,
    });
  });

  app.post('/api/admin/security/rotate-totp', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    if (!store.adminCredential) {
      res.status(400).json({ error: 'Claimed admin credentials required.' });
      return;
    }
    const currentPassword = String(req.body?.currentPassword ?? '');
    const computed = hashPassword(currentPassword, store.adminCredential.salt);
    if (!safeCompareString(computed, store.adminCredential.passwordHash)) {
      res.status(401).json({ error: 'Current password is required to rotate authenticator secret.' });
      return;
    }
    const secret = generateTotpSecret();
    store.adminCredential.totpSecret = secret;
    store.adminCredential.updatedAt = new Date().toISOString();
    saveStore(store);
    const otpauthUri = buildOtpAuthUri({
      issuer: 'Thinker Admin',
      accountName: store.adminCredential.email || store.adminCredential.username,
      secret,
    });
    res.json({
      totpSetup: {
        secret,
        otpauthUri,
        issuer: 'Thinker Admin',
        accountName: store.adminCredential.email || store.adminCredential.username,
      },
    });
  });

  app.post('/api/admin/security/verify-totp', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    if (!store.adminCredential) {
      res.status(400).json({ error: 'Claimed admin credentials required.' });
      return;
    }
    const code = String(req.body?.code ?? '').trim();
    const valid = verifyTotpCode(store.adminCredential.totpSecret, code);
    if (!valid) {
      res.status(400).json({ error: 'Code does not match current 30-second window. Check device clock.' });
      return;
    }
    res.json({ valid: true, message: '30-second authenticator code verified cleanly.' });
  });

  // Protected Admin Dashboard Routes
  app.get('/api/admin/analytics', requireAdmin, (_req: Request, res: Response) => {
    const store = ensureStore();
    const summary = aggregateAnalyticsSummary(
      store.events,
      store.siteConfig.projects,
      store.contacts.length,
      store.reviews
    );
    const cred = store.adminCredential;
    res.json({
      summary,
      security: store.security,
      securityInfo: cred
        ? ({
            email: cred.email,
            totpEnabled: Boolean(cred.totpSecret),
            backupCodesRemaining: cred.backupCodeHashes.length,
            cooldownHours: cred.cooldownHours,
            maxFailedAttempts: cred.maxFailedAttempts,
            totpSecondsRemaining: getTotpSecondsRemaining(),
          } satisfies AdminSecurityInfo)
        : undefined,
      adminSource: getAdminStatus(store).source,
      contacts: store.contacts,
      reviews: store.reviews,
      questions: store.questions,
      siteConfig: store.siteConfig,
    });
  });

  app.put('/api/admin/site-config', requireAdmin, (req: Request, res: Response) => {
    const incoming = req.body?.siteConfig as Partial<SiteConfig> | undefined;
    if (!incoming || typeof incoming !== 'object') {
      res.status(400).json({ error: 'Missing siteConfig payload.' });
      return;
    }
    const store = ensureStore();
    const merged = mergeSiteConfig(INITIAL_SITE_CONFIG, incoming);

    merged.brand.logoUrl =
      sanitizeExternalUrl(merged.brand.logoUrl, 'image') || store.siteConfig.brand.logoUrl;
    merged.hero.heroImageUrl =
      sanitizeExternalUrl(merged.hero.heroImageUrl, 'image') || store.siteConfig.hero.heroImageUrl;
    merged.about.emblemUrl =
      sanitizeExternalUrl(merged.about.emblemUrl, 'image') || store.siteConfig.about.emblemUrl;
    merged.disciplines.items = merged.disciplines.items.slice(0, 8).map((item) => ({
      title: String(item.title || '').trim().slice(0, 80),
      desc: String(item.desc || '').trim().slice(0, 260),
      symbolUrl: sanitizeExternalUrl(item.symbolUrl, 'image') || '/assets/aistudio/26254-directory_dcde0850.png',
    }));
    merged.about.journeySteps = merged.about.journeySteps.slice(0, 8).map((step) => ({
      period: String(step.period || '').trim().slice(0, 60),
      title: String(step.title || '').trim().slice(0, 100),
      detail: String(step.detail || '').trim().slice(0, 320),
      symbolUrl: sanitizeExternalUrl(step.symbolUrl, 'image') || '/assets/aistudio/Partner_b2d8c81e.png',
    }));
    merged.footer.socials = {
      discordUrl: sanitizeExternalUrl(merged.footer.socials.discordUrl, 'discord'),
      discordLabel: String(merged.footer.socials.discordLabel || 'Discord').trim().slice(0, 40),
      githubUrl: sanitizeExternalUrl(merged.footer.socials.githubUrl, 'github'),
      githubLabel: String(merged.footer.socials.githubLabel || 'GitHub').trim().slice(0, 40),
      twitterUrl: sanitizeExternalUrl(merged.footer.socials.twitterUrl, 'twitter'),
      twitterLabel: String(merged.footer.socials.twitterLabel || 'X / Twitter').trim().slice(0, 40),
    };
    const minBudget = Math.max(1, Math.min(10000, Math.round(Number(merged.contactConfig.minCustomBudget) || 5)));
    const maxBudget = Math.max(minBudget, Math.min(50000, Math.round(Number(merged.contactConfig.maxCustomBudget) || 30)));
    merged.contactConfig = {
      defaultBudget: String(merged.contactConfig.defaultBudget || `$${minBudget} - $${maxBudget}`).trim().slice(0, 60),
      budgetPresets: merged.contactConfig.budgetPresets
        .map((b) => String(b || '').trim().slice(0, 60))
        .filter(Boolean)
        .slice(0, 8),
      minCustomBudget: minBudget,
      maxCustomBudget: maxBudget,
      allowCustomBudget: Boolean(merged.contactConfig.allowCustomBudget),
      serviceScopes: merged.contactConfig.serviceScopes
        .map((s) => String(s || '').trim().slice(0, 80))
        .filter(Boolean)
        .slice(0, 10),
    };
    merged.projects = merged.projects.slice(0, 30).map((p) => sanitizeProjectPayload(p));

    store.siteConfig = merged;
    saveStore(store);
    res.json({ siteConfig: store.siteConfig });
  });

  app.post('/api/admin/site-config/reset', requireAdmin, (_req: Request, res: Response) => {
    const store = ensureStore();
    store.siteConfig = cloneDefaultSiteConfig();
    saveStore(store);
    res.json({ siteConfig: store.siteConfig });
  });

  app.patch('/api/admin/contacts/:id', requireAdmin, (req: Request, res: Response) => {
    const { id } = req.params;
    const store = ensureStore();
    const item = store.contacts.find((c) => c.id === id);
    if (!item) {
      res.status(404).json({ error: 'Message not found.' });
      return;
    }
    if (req.body?.status !== undefined) {
      const status = String(req.body.status);
      if (!['unread', 'in-progress', 'replied', 'archived'].includes(status)) {
        res.status(400).json({ error: 'Invalid contact status.' });
        return;
      }
      item.status = status as ContactSubmission['status'];
    }
    if (req.body?.category !== undefined) {
      const cat = String(req.body.category) as InquiryCategory;
      if (ALLOWED_INQUIRY_CATEGORIES.has(cat)) {
        item.category = cat;
      }
    }
    if (req.body?.adminNotes !== undefined) {
      item.adminNotes = String(req.body.adminNotes).trim().slice(0, 1000);
    }
    saveStore(store);
    res.json({ contact: item });
  });

  app.delete('/api/admin/contacts/:id', requireAdmin, (req: Request, res: Response) => {
    const { id } = req.params;
    const store = ensureStore();
    const before = store.contacts.length;
    store.contacts = store.contacts.filter((c) => c.id !== id);
    if (store.contacts.length === before) {
      res.status(404).json({ error: 'Message not found.' });
      return;
    }
    saveStore(store);
    res.json({ ok: true });
  });

  app.delete('/api/reviews/:id', requireAdmin, (req: Request, res: Response) => {
    const { id } = req.params;
    const store = ensureStore();
    const before = store.reviews.length;
    store.reviews = store.reviews.filter((r) => r.id !== id);
    if (store.reviews.length === before) {
      res.status(404).json({ error: 'Review not found.' });
      return;
    }
    saveStore(store);
    res.json({ ok: true });
  });

  // Admin Q&A Moderation Routes
  app.patch('/api/admin/questions/:id', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    const q = store.questions.find((item) => item.id === req.params.id);
    if (!q) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    if (typeof req.body?.isPinned === 'boolean') {
      q.isPinned = req.body.isPinned;
    }
    if (typeof req.body?.acceptedAnswerId === 'string') {
      const targetId = req.body.acceptedAnswerId;
      q.answers = q.answers.map((a) => ({
        ...a,
        isAccepted: a.id === targetId ? !a.isAccepted : false,
      }));
    }
    saveStore(store);
    res.json({ question: q });
  });

  app.delete('/api/admin/questions/:id', requireAdmin, (req: Request, res: Response) => {
    const store = ensureStore();
    const before = store.questions.length;
    store.questions = store.questions.filter((q) => q.id !== req.params.id);
    if (store.questions.length === before) {
      res.status(404).json({ error: 'Question not found.' });
      return;
    }
    saveStore(store);
    res.json({ ok: true });
  });

  app.delete(
    '/api/admin/questions/:id/answers/:answerId',
    requireAdmin,
    (req: Request, res: Response) => {
      const store = ensureStore();
      const q = store.questions.find((item) => item.id === req.params.id);
      if (!q) {
        res.status(404).json({ error: 'Question not found.' });
        return;
      }
      const before = q.answers.length;
      q.answers = q.answers.filter((a) => a.id !== req.params.answerId);
      if (q.answers.length === before) {
        res.status(404).json({ error: 'Answer not found.' });
        return;
      }
      saveStore(store);
      res.json({ question: q });
    }
  );

  // Server-Side Gemini Creator Guide & Architecture Advisor
  app.post('/api/ai/guide', async (req: Request, res: Response) => {
    if (!enforceRateLimit(aiRateLimiter, 'ai-guide', req, res)) return;

    const prompt = String(req.body?.prompt ?? '').trim().slice(0, 1500);
    const mode = req.body?.mode === 'deep' ? 'deep' : 'fast';
    const sessionId = String(req.body?.sessionId ?? 'anon').slice(0, 64);

    if (prompt.length < 4) {
      res.status(400).json({ error: 'Please enter a question or server architecture prompt.' });
      return;
    }

    const ai = getAiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'Gemini API key is not configured yet. Open Settings > Secrets in AI Studio to attach a valid API key.',
      });
      return;
    }

    try {
      const store = ensureStore();
      store.events.push({
        id: `ev-${crypto.randomUUID().slice(0, 8)}`,
        type: 'guide_query',
        sessionId,
        target: mode,
        timestamp: new Date().toISOString(),
      });
      saveStore(store);

      const systemInstruction =
        mode === 'deep'
          ? 'You are Thinker, a senior Discord Server Developer & Creator Guide. Provide a structured, practical architectural blueprint covering channel hierarchy, role permissions, onboarding flow, and AutoMod safeguards. Keep prose crisp and direct.'
          : 'You are Thinker, a Discord Server Developer & Creator Guide. Give a concise, actionable, 3-5 sentence answer focused on Discord community architecture, onboarding, or Thinker’s capabilities.';

      if (mode === 'deep') {
        const response = await ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: prompt,
          config: {
            systemInstruction,
            thinkingConfig: {
              thinkingLevel: ThinkingLevel.HIGH,
            },
          },
        });
        res.json({
          text: response.text ?? '',
          modelUsed: 'gemini-4-hydrogen',
          mode,
        });
      } else {
        const response = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: prompt,
          config: {
            systemInstruction,
          },
        });
        res.json({
          text: response.text ?? '',
          modelUsed: 'gemini-3.1-flash-lite',
          mode,
        });
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Unable to reach Gemini API right now.';
      res.status(502).json({ error: message });
    }
  });

  // Server-Side AI Project Brief Assistant (helps prospective clients structure inquiries in Contact)
  app.post('/api/ai/brief-assistant', async (req: Request, res: Response) => {
    if (!enforceRateLimit(aiRateLimiter, 'ai-brief', req, res)) return;

    const notes = String(req.body?.notes ?? '').trim().slice(0, 1200);
    const serviceScope = String(req.body?.serviceScope ?? 'Full Discord Server Build').trim().slice(0, 100);
    const serverSize = String(req.body?.serverSize ?? 'Launching / < 500 members').trim().slice(0, 60);
    const timeline = String(req.body?.timeline ?? 'Within 2-4 weeks').trim().slice(0, 60);
    const sessionId = String(req.body?.sessionId ?? 'anon').slice(0, 64);

    if (notes.length < 6) {
      res.status(400).json({ error: 'Enter at least a short sentence describing what your Discord server needs.' });
      return;
    }

    const ai = getAiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'Gemini API key is not configured yet. Open Settings > Secrets in AI Studio to attach a valid API key.',
      });
      return;
    }

    try {
      const store = ensureStore();
      store.events.push({
        id: `ev-${crypto.randomUUID().slice(0, 8)}`,
        type: 'guide_query',
        sessionId,
        target: 'brief-assistant',
        timestamp: new Date().toISOString(),
      });
      saveStore(store);

      const prompt = [
        `Client inquiry context:`,
        `- Scope: ${serviceScope}`,
        `- Server Stage/Size: ${serverSize}`,
        `- Target Timeline: ${timeline}`,
        `- Rough Notes: "${notes}"`,
        ``,
        `Write a clean, ready-to-send 1-paragraph project brief in first-person ("We are...") that the client can paste into Thinker's contact form, followed by a short bulleted "Recommended Setup Checklist" (3 concise items) tailored to Discord server architecture.`,
      ].join('\n');

      const response = await ai.models.generateContent({
        model: 'gemini-3.1-flash-lite',
        contents: prompt,
        config: {
          systemInstruction:
            'You help prospective Discord community owners turn rough ideas into clear, concise project inquiry briefs for Thinker (Discord Server Developer & Creator Guide). Be practical, specific, and brief.',
        },
      });

      res.json({
        draft: response.text ?? '',
        modelUsed: 'gemini-3.1-flash-lite',
      });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Unable to reach Gemini API right now.';
      res.status(502).json({ error: message });
    }
  });

  // Server-Side AI Q&A Discussion Helper (drafts Discord architecture advice for a Q&A thread)
  app.post('/api/ai/qa-assist', async (req: Request, res: Response) => {
    if (!enforceRateLimit(aiRateLimiter, 'ai-qa', req, res)) return;

    const title = String(req.body?.title ?? '').trim().slice(0, 200);
    const questionBody = String(req.body?.body ?? '').trim().slice(0, 1500);
    const category = String(req.body?.category ?? 'Server Setup').trim().slice(0, 60);
    const sessionId = String(req.body?.sessionId ?? 'anon').slice(0, 64);

    if (title.length < 5) {
      res.status(400).json({ error: 'Question context is too short.' });
      return;
    }

    const ai = getAiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'Gemini API key is not configured yet. Open Settings > Secrets in AI Studio to attach a valid API key.',
      });
      return;
    }

    try {
      const store = ensureStore();
      store.events.push({
        id: `ev-${crypto.randomUUID().slice(0, 8)}`,
        type: 'guide_query',
        sessionId,
        target: 'qa-assist',
        timestamp: new Date().toISOString(),
      });
      saveStore(store);

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Category: ${category}\nQuestion: ${title}\nDetails: ${questionBody}`,
        config: {
          systemInstruction:
            'You are Thinker, a Discord Server Developer & Creator Guide. Draft a concise, directly actionable answer (4-6 sentences) covering exact Discord channels, Onboarding, role permission hygiene, or AutoMod steps. No fluff.',
          thinkingConfig: {
            thinkingLevel: ThinkingLevel.LOW,
          },
        },
      });

      res.json({
        suggestion: response.text ?? '',
        modelUsed: 'gemini-4-hydrogen',
      });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Unable to reach Gemini API right now.';
      res.status(502).json({ error: message });
    }
  });

  // Admin AI Client Reply Draft Helper (concise inquiry response outline)
  app.post('/api/ai/admin-reply-draft', requireAdmin, async (req: Request, res: Response) => {
    if (!enforceRateLimit(aiRateLimiter, 'ai-admin-reply', req, res)) return;

    const clientName = String(req.body?.name ?? 'Client').trim().slice(0, 80);
    const serviceScope = String(req.body?.serviceScope ?? 'Discord Server Build').trim().slice(0, 100);
    const budgetRange = String(req.body?.budgetRange ?? '$5 - $30').trim().slice(0, 60);
    const message = String(req.body?.message ?? '').trim().slice(0, 1500);

    if (message.length < 4) {
      res.status(400).json({ error: 'Client message is too short to draft a reply.' });
      return;
    }

    const ai = getAiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'Gemini API key is not configured yet. Open Settings > Secrets in AI Studio to attach a valid API key.',
      });
      return;
    }

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-flash-lite',
        contents: `Client: ${clientName}\nScope: ${serviceScope}\nBudget: ${budgetRange}\nMessage: ${message}`,
        config: {
          systemInstruction:
            'You are Thinker, a Discord Server Developer & Creator Guide. Write a warm, concise, 3-5 sentence client reply acknowledging their scope & budget, proposing next steps, and asking 1-2 clarifying questions.',
        },
      });
      res.json({
        draft: response.text ?? '',
        modelUsed: 'gemini-3.1-flash-lite',
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unable to reach Gemini API right now.';
      res.status(502).json({ error: msg });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  // Graceful JSON error handler
  app.use((err: unknown, _req: Request, res: Response, next: NextFunction) => {
    if (res.headersSent) {
      next(err);
      return;
    }
    const httpErr = err as { type?: string; status?: number; message?: string } | null;
    if (httpErr?.type === 'entity.too.large' || httpErr?.status === 413) {
      res.status(413).json({
        error: 'Request payload is too large. Please compress or use smaller images.',
      });
      return;
    }
    if (err instanceof SyntaxError && httpErr?.status === 400) {
      res.status(400).json({ error: 'Malformed JSON request body.' });
      return;
    }
    res.status(httpErr?.status || 500).json({ error: 'Internal server error.' });
  });

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Thinker portfolio server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
