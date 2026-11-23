import type { SVGProps } from 'react'

type Props = SVGProps<SVGSVGElement> & { size?: number }

function Icon({ size = 18, children, ...props }: Props) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{children}</svg>
}

export const PanelIcon = (props: Props) => <Icon {...props}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></Icon>
export const DemoIcon = (props: Props) => <Icon {...props}><path d="M21 11.5a8.4 8.4 0 0 1-9 8.5 9.2 9.2 0 0 1-4-.9L3 20l1.1-4.3A8.5 8.5 0 1 1 21 11.5Z"/><path d="M8 12h.01M12 12h.01M16 12h.01" strokeWidth="2.4"/></Icon>
export const ChartIcon = (props: Props) => <Icon {...props}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></Icon>
export const ModelIcon = (props: Props) => <Icon {...props}><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6v6H9zM9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></Icon>
export const SettingsIcon = (props: Props) => <Icon {...props}><path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/></Icon>
export const ChevronDown = (props: Props) => <Icon {...props}><path d="m7 9.5 5 5 5-5"/></Icon>
export const CheckIcon = (props: Props) => <Icon {...props}><path d="m5 12 4 4L19 6"/></Icon>
export const SendIcon = (props: Props) => <Icon {...props}><path d="M12 19V5M6.5 10.5 12 5l5.5 5.5"/></Icon>
export const CloseIcon = (props: Props) => <Icon {...props}><path d="m6 6 12 12M18 6 6 18"/></Icon>
export const ExpandIcon = (props: Props) => <Icon {...props}><path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5M3 3l6 6M21 3l-6 6M3 21l6-6M21 21l-6-6"/></Icon>
export const PlusIcon = (props: Props) => <Icon {...props}><path d="M12 5v14M5 12h14"/></Icon>
export const MinusIcon = (props: Props) => <Icon {...props}><path d="M5 12h14"/></Icon>
export const WarningIcon = (props: Props) => <Icon {...props}><path d="M10.3 3.6 2.5 17a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.6a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01" strokeWidth="2"/></Icon>
