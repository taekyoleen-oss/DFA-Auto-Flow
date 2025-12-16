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
} from "@heroicons/react/24/outline";

// Solid icons (if needed)
import {
  CheckCircleIcon,
} from "@heroicons/react/24/solid";

// Custom icons that might not be in Heroicons
// LogoIcon - 간단한 SVG로 정의
export const LogoIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
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
      d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
    />
  </svg>
);

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
  // Aliased icons
  BarChartIcon,
  PriceTagIcon,
  FilterIcon,
  // Custom icons (already exported above as const)
};
