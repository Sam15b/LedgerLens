"use client";

import { usePathname } from "next/navigation";

import { NavList } from "@/components/NavList";

/** The links with the current page marked. The path is request data: render this inside
 * <Suspense> (a dynamic route such as /documents/[id] cannot be prerendered otherwise). */
export function ActiveNavList() {
  return <NavList pathname={usePathname()} />;
}
