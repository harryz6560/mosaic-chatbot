import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef(({ className, ...props }, ref) => {
  return (
    <textarea
      rows={1}
      ref={ref}
      className={cn(
        "w-full resize-none bg-transparent px-4 py-3 text-sm text-white placeholder-gray-400 focus:outline-none disabled:opacity-50",
        className
      )}
      {...props}
    />
  );
});
Input.displayName = "Input"

export { Input }
