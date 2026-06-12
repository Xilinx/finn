/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the defintion of the SMBus FInite State Machine
 *
 * @file smbus_state.h
 *
 */

#ifndef _SMBUS_STATE_H_
#define _SMBUS_STATE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "smbus.h"

/******************************************************************************
*
* @brief    Is the finite state machine for the specified SMBus instance
*           The function will look up the current state, bytes sent, bytes received etc 
*           and given the event passed in it will transition to a new state 
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusFSM( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Is a conversion function from the state enum to character string
*           to be used by logging functions
*
* @param    xState is any of the state machine state values
*
* @return   A character string
*
* @note     None.
*
*****************************************************************************/
char* pcStateToString( uint8_t xState );

#ifdef __cplusplus
}
#endif

#endif /* _SMBUS_STATE_H_ */
