import React from "react";
// Heroicons에서 필요한 아이콘들을 re-export
import {
  // Outline icons
  PlayIcon,
  CodeBracketIcon,
  FolderOpenIcon,
  PlusIcon,
  MinusIcon,
  Bars3Icon,
  CogIcon,
  ArrowUturnLeftIcon,
  ArrowUturnRightIcon,
  SparklesIcon,
  ArrowsPointingOutIcon,
  Squares2X2Icon,
  CheckIcon,
  ArrowPathIcon,
  StarIcon,
  XCircleIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  XMarkIcon,
  ClipboardIcon,
  TableCellsIcon,
  CommandLineIcon,
  InformationCircleIcon,
  LinkIcon,
  DocumentTextIcon,
  RectangleStackIcon,
  CircleStackIcon,
  ChartBarIcon as BarChartIcon,
  ShareIcon,
  CheckBadgeIcon,
  CalculatorIcon,
  TagIcon as PriceTagIcon,
  FunnelIcon as FilterIcon,
  UsersIcon,
  BeakerIcon,
  HashtagIcon,
  PresentationChartLineIcon,
  ShieldCheckIcon,
  ChartPieIcon,
  FingerPrintIcon,
  ArrowDownTrayIcon,
  SunIcon,
  MoonIcon,
} from "@heroicons/react/24/outline";

// Solid icons (if needed)
import {
  CheckCircleIcon,
} from "@heroicons/react/24/solid";

// Custom icons that might not be in Heroicons

// DatabaseIcon - Heroicons에 없으므로 직접 정의
export const DatabaseIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"
    />
  </svg>
);

// ScaleIcon - Heroicons에 없으므로 직접 정의
export const ScaleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.224 48.224 0 0012 4.5c-2.48 0-4.785.685-6.75 1.97m13.5 0c1.01.943 1.902 2.12 2.6 3.5m-13.5 0c1.01-.943 1.902-2.12 2.6-3.5m15.75 0c-1.01.943-1.902 2.12-2.6 3.5m-15.75 0c1.01-.943 1.902-2.12 2.6-3.5"
    />
  </svg>
);

// BellCurveIcon - Heroicons에 없으므로 직접 정의
export const BellCurveIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M3 13.5L7.5 8.25 12 13.5l4.5-4.5L21 13.5M3 21h18M12 3v18"
    />
  </svg>
);

// ChartCurveIcon - Heroicons에 없으므로 직접 정의
export const ChartCurveIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.94"
    />
  </svg>
);

// FontSizeIncreaseIcon
export const FontSizeIncreaseIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25H12"
    />
  </svg>
);

// FontSizeDecreaseIcon
export const FontSizeDecreaseIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
    />
  </svg>
);

// Export all icons
export {
  PlayIcon,
  CodeBracketIcon,
  FolderOpenIcon,
  PlusIcon,
  MinusIcon,
  Bars3Icon,
  CogIcon,
  ArrowUturnLeftIcon,
  ArrowUturnRightIcon,
  SparklesIcon,
  ArrowsPointingOutIcon,
  Squares2X2Icon,
  CheckIcon,
  ArrowPathIcon,
  StarIcon,
  XCircleIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  XMarkIcon,
  ClipboardIcon,
  TableCellsIcon,
  CommandLineIcon,
  InformationCircleIcon,
  LinkIcon,
  DocumentTextIcon,
  RectangleStackIcon,
  CircleStackIcon,
  ShareIcon,
  CheckBadgeIcon,
  CalculatorIcon,
  UsersIcon,
  BeakerIcon,
  HashtagIcon,
  PresentationChartLineIcon,
  ShieldCheckIcon,
  ChartPieIcon,
  FingerPrintIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon,
  SunIcon,
  MoonIcon,
  // Aliased icons
  BarChartIcon,
  PriceTagIcon,
  FilterIcon,
  // Custom icons (already exported above as const)
};

/**
 * tkLeen 브랜드 마크 (tkLeen-BRAND-GUIDE.md 정본 지오메트리, viewBox 0 0 200 200).
 * T = Ink(#0A0A0A, 밝은 배경) / Cream(#FAFAF7, 어두운 배경)
 * k = Sky Blue(#4A90C2) — 배경과 무관하게 고정(브랜드 시그니처, 변경 금지)
 */
export const LogoIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    viewBox="0 0 200 200"
    className={className}
    role="img"
    aria-label="tkLeen"
    xmlns="http://www.w3.org/2000/svg"
  >
    {/* T — 배경에 따라 Ink / Cream */}
    <rect x="40" y="40" width="80" height="20" className="fill-[#0A0A0A] dark:fill-[#FAFAF7]" />
    <rect x="60" y="60" width="20" height="40" className="fill-[#0A0A0A] dark:fill-[#FAFAF7]" />
    {/* k — Sky Blue 고정 */}
    <rect x="60" y="100" width="20" height="60" fill="#4A90C2" />
    <rect x="80" y="100" width="20" height="20" fill="#4A90C2" />
    <rect x="100" y="80" width="20" height="20" fill="#4A90C2" />
    <rect x="120" y="60" width="20" height="20" fill="#4A90C2" />
    <rect x="80" y="120" width="20" height="20" fill="#4A90C2" />
    <rect x="100" y="140" width="20" height="20" fill="#4A90C2" />
    <rect x="120" y="160" width="20" height="20" fill="#4A90C2" />
  </svg>
);
