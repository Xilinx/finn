/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the structs and function declarations for the event buffer functions
 *
 * @file smbus_event_buffer.h
 *
 */

#ifndef _SMBUS_EVENT_BUFFER_H_
#define _SMBUS_EVENT_BUFFER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "smbus.h"

#define SMBUS_EVENT_BUFFER_SUCCESS          ( 0 )
#define SMBUS_EVENT_BUFFER_FAIL             ( 1 )

/******************************************************************************
*
* @brief    Walks through all elements of the event buffer. It sets all the elements to unoccupied  
*           and sets the event value to 0.
*
* @param    pxContext is a pointer to the event buffer structure.
* @param    pxEventBuffer is the pointer to the event buffer element
* @param    ulMaxElements is the number of elements in the event log.
*
* @return   None     
*
* @note     None.
*
*****************************************************************************/
void vEventBufferInitialize( SMBUS_EVENT_BUFFER_TYPE* pxContext, SMBUS_EVENT_BUFFER_ELEMENT_TYPE* pxEventBuffer, uint32_t ulMaxElements );

/******************************************************************************
*
* @brief    If the write location is not occupied the event is written to that
*           location. The location is marked as occupied and the write location
*           is incremented
*
* @param    pxContext is a pointer to the event buffer structure.
* @param    ucAnyCharacter is the event to write to the event buffer
* @param    pulWrite_Position is a pointer to the location the event was written to
*
* @return   SMBUS_FALSE - if write fails
*           SMBUS_TRUE  - if write succeeds       
*
* @note     None.
*
*****************************************************************************/
uint8_t ucEventBufferTryWrite( SMBUS_EVENT_BUFFER_TYPE* pxContext, uint8_t ucAnyCharacter, uint32_t* pulWrite_Position );

/******************************************************************************
*
* @brief    If the read location is occupied the event at the location is read
*           The read location is then marked as empty and the read location incremented
*
* @param    pxContext is a pointer to the event buffer structure.
* @param    pucAnyCharacter is a pointer to the event read
* @param    pulRead_Position is a pointer to the location the event was read from
*
* @return   SMBUS_FALSE - if read fails
*           SMBUS_TRUE  - if write succeeds
*
* @note     None.
*
*****************************************************************************/
uint8_t ucEventBufferTryRead( SMBUS_EVENT_BUFFER_TYPE* pxContext, uint8_t* pucAnyCharacter, uint32_t* pulRead_Position );

#ifdef __cplusplus
}
#endif

#endif /* _SMBUS_EVENT_BUFFER_H_ */
