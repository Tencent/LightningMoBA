/**
 * @file lightning_moba_namespace_config.h
 * @brief Configuration file for Flash namespace management and isolation
 *
 * This header provides configuration macros for managing the Flash namespace
 * across a codebase. It allows for flexible namespace naming and provides
 * utilities for namespace declaration and scoping.
 *
 * Usage Examples:
 *
 * 1. Basic namespace wrapping:
 * @code
 *   BEGIN_LIGHTNING_MOBA_NAMESPACE
 *   class FlashDevice {
 *     // Implementation
 *   };
 *   END_LIGHTNING_MOBA_NAMESPACE
 * @endcode
 *
 * 2. Accessing types within the namespace:
 * @code
 *   LIGHTNING_MOBA_NAMESPACE_ALIAS(FlashDevice) device;
 * @endcode
 *
 * 3. Defining content within namespace scope:
 * @code
 *   LIGHTNING_MOBA_NAMESPACE_SCOPE(
 *     struct Configuration {
 *       uint32_t size;
 *       bool enabled;
 *     };
 *   )
 * @endcode
 *
 * 4. Custom namespace name:
 * @code
 *   #define LIGHTNING_MOBA_NAMESPACE custom_flash
 *   #include "LIGHTNING_MOBA_NAMESPACE_config.h"
 * @endcode
 *
 * Configuration:
 * - The default namespace is 'flash' if LIGHTNING_MOBA_NAMESPACE is not defined
 * - Define LIGHTNING_MOBA_NAMESPACE before including this header to customize the
 * namespace name
 *
 * Best Practices:
 * - Include this header in all files that need access to the Flash namespace
 *
 */
#pragma once

#ifndef LIGHTNING_MOBA_NAMESPACE_CONFIG_H
#define LIGHTNING_MOBA_NAMESPACE_CONFIG_H

// Set default namespace to flash
#ifndef LIGHTNING_MOBA_NAMESPACE
#define LIGHTNING_MOBA_NAMESPACE flash
#endif

#define LIGHTNING_MOBA_NAMESPACE_ALIAS(name) LIGHTNING_MOBA_NAMESPACE::name

#define LIGHTNING_MOBA_NAMESPACE_SCOPE(content)                                         \
  namespace LIGHTNING_MOBA_NAMESPACE {                                                  \
  content                                                                      \
  }

#endif // LIGHTNING_MOBA_NAMESPACE_CONFIG_H
